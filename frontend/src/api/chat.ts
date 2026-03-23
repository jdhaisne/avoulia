const API_BASE = import.meta.env.VITE_API_URL ?? '/api/v1'

/** Message d'accueil affiché au chargement du chat (fallback si l'API échoue). */
export const WELCOME_MESSAGE_FALLBACK =
  "Bonjour, je vais vous aider à identifier des cas d'usage concrets de l'IA adaptés à votre organisation. Pour commencer, je vais vous poser quelques questions simples afin de cibler précisément votre priorité."

export interface WelcomeResponse {
  message: string
}

export async function getWelcomeMessage(): Promise<string> {
  try {
    const url = `${API_BASE}/chat/welcome`
    const res = await fetch(url)
    if (!res.ok) throw new Error('Impossible de charger le message d\'accueil')
    const data: WelcomeResponse = await res.json()
    return data.message ?? WELCOME_MESSAGE_FALLBACK
  } catch {
    return WELCOME_MESSAGE_FALLBACK
  }
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

/** Un cas suggéré (id + contenu), à renvoyer dans last_suggested_cases pour le détail. */
export interface SuggestedCase {
  id: string
  content: string
}

export interface ChatRequest {
  message: string
  history: ChatMessage[]
  /** Liste des cas proposés au tour précédent (pour « détaille le 2 » / « ok vas-y »). */
  last_suggested_cases?: SuggestedCase[] | null
  pending_action?: string | null
  pending_use_case_id?: string | null
  selected_domain_code?: string | null
  selected_sector?: string | null
  selected_intention?: string | null
}

export interface ChatResponse {
  answer: string
  sources: string[]
}

export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  const url = `${API_BASE}/chat`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Erreur lors de l'envoi du message")
  }
  return res.json()
}

/** Payload envoyé à la fin du stream (data: { done: true, ... }). */
export interface StreamDonePayload {
  sources: string[]
  suggested_cases?: SuggestedCase[]
  suggested_case_ids?: string[]
  selected_domain_code?: string | null
  selected_sector?: string | null
  selected_intention?: string | null
  pending_action?: string | null
  pending_use_case_id?: string | null
  pending_case_index?: number | null
}

export interface StreamCallbacks {
  onToken: (token: string) => void
  onDone: (payload: StreamDonePayload) => void
  onError: (message: string) => void
}

export async function sendMessageStream(
  request: ChatRequest,
  callbacks: StreamCallbacks
): Promise<void> {
  const url = `${API_BASE}/chat/stream`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    callbacks.onError(err.detail || "Erreur lors de l'envoi du message")
    return
  }
  const reader = res.body?.getReader()
  if (!reader) {
    callbacks.onError('Stream non disponible')
    return
  }
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim()
          if (!raw) continue
          try {
            const data = JSON.parse(raw) as {
              t?: string
              done?: boolean
              sources?: string[]
              suggested_cases?: SuggestedCase[]
              suggested_case_ids?: string[]
              selected_domain_code?: string | null
              selected_sector?: string | null
              selected_intention?: string | null
              pending_action?: string | null
              pending_use_case_id?: string | null
              pending_case_index?: number | null
              error?: string
            }
            if (data.error) {
              callbacks.onError(data.error)
              return
            }
            if (data.t) callbacks.onToken(data.t)
            if (data.done === true) {
              callbacks.onDone({
                sources: data.sources ?? [],
                suggested_cases: data.suggested_cases,
                suggested_case_ids: data.suggested_case_ids,
                selected_domain_code: data.selected_domain_code ?? null,
                selected_sector: data.selected_sector ?? null,
                selected_intention: data.selected_intention ?? null,
                pending_action: data.pending_action ?? null,
                pending_use_case_id: data.pending_use_case_id ?? null,
                pending_case_index: data.pending_case_index ?? null,
              })
              return
            }
          } catch {
            // ignore malformed line
          }
        }
      }
    }
    if (buffer.startsWith('data: ')) {
      try {
        const data = JSON.parse(buffer.slice(6).trim()) as {
          done?: boolean
          sources?: string[]
          suggested_cases?: SuggestedCase[]
          suggested_case_ids?: string[]
          selected_domain_code?: string | null
          selected_sector?: string | null
          selected_intention?: string | null
          pending_action?: string | null
          pending_use_case_id?: string | null
          pending_case_index?: number | null
        }
        if (data.done === true) {
          callbacks.onDone({
            sources: data.sources ?? [],
            suggested_cases: data.suggested_cases,
            suggested_case_ids: data.suggested_case_ids,
            selected_domain_code: data.selected_domain_code ?? null,
            selected_sector: data.selected_sector ?? null,
            selected_intention: data.selected_intention ?? null,
            pending_action: data.pending_action ?? null,
            pending_use_case_id: data.pending_use_case_id ?? null,
            pending_case_index: data.pending_case_index ?? null,
          })
        } else {
          callbacks.onDone({ sources: [] })
        }
      } catch {
        callbacks.onDone({ sources: [] })
      }
    } else {
      callbacks.onDone({ sources: [] })
    }
  } catch (e) {
    callbacks.onError(e instanceof Error ? e.message : 'Erreur stream')
  }
}
