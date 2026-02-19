const API_BASE = import.meta.env.VITE_API_URL ?? '/api/v1'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  message: string
  history: ChatMessage[]
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

export interface StreamCallbacks {
  onToken: (token: string) => void
  onDone: (sources: string[]) => void
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
            const data = JSON.parse(raw) as { t?: string; done?: boolean; sources?: string[]; error?: string }
            if (data.error) {
              callbacks.onError(data.error)
              return
            }
            if (data.t) callbacks.onToken(data.t)
            if (data.done === true) {
              callbacks.onDone(data.sources ?? [])
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
        const data = JSON.parse(buffer.slice(6).trim()) as { done?: boolean; sources?: string[] }
        if (data.done === true) callbacks.onDone(data.sources ?? [])
      } catch {
        callbacks.onDone([])
      }
    } else {
      callbacks.onDone([])
    }
  } catch (e) {
    callbacks.onError(e instanceof Error ? e.message : 'Erreur stream')
  }
}
