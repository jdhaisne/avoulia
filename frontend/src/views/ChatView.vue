<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import type { ChatMessage, SuggestedCase } from '@/api/chat'
import { sendMessageStream, getWelcomeMessage } from '@/api/chat'
import microsoftLogo from '@/assets/microsoft-logo.svg'

const messages = ref<ChatMessage[]>([])
const input = ref('')
const loading = ref(false)
const error = ref<string | null>(null)
const messagesEnd = ref<HTMLDivElement | null>(null)
const chatInputRef = ref<HTMLInputElement | null>(null)

// État renvoyé par le backend à chaque réponse, pour la requête suivante (détail « le 2 », ok/vas-y, domaine/secteur)
const lastSuggestedCases = ref<SuggestedCase[] | null>(null)
const selectedDomainCode = ref<string | null>(null)
const selectedSector = ref<string | null>(null)
const selectedIntention = ref<string | null>(null)
const pendingAction = ref<string | null>(null)
const pendingUseCaseId = ref<string | null>(null)

function scrollToBottom() {
  nextTick(() => {
    messagesEnd.value?.scrollIntoView({ behavior: 'smooth' })
  })
}

onMounted(async () => {
  if (messages.value.length === 0) {
    const welcome = await getWelcomeMessage()
    messages.value = [{ role: 'assistant', content: welcome }]
    nextTick(() => scrollToBottom())
  }
  nextTick(() => chatInputRef.value?.focus())
})

async function submit() {
  const text = input.value.trim()
  if (!text || loading.value) return
  input.value = ''
  error.value = null
  const userMessage: ChatMessage = { role: 'user', content: text }
  messages.value = [...messages.value, userMessage]
  messages.value = [...messages.value, { role: 'assistant', content: '' }]
  await nextTick()
  scrollToBottom()
  loading.value = true
  try {
    await sendMessageStream(
      {
        message: text,
        history: messages.value.slice(0, -2),
        last_suggested_cases: lastSuggestedCases.value ?? undefined,
        pending_action: pendingAction.value ?? undefined,
        pending_use_case_id: pendingUseCaseId.value ?? undefined,
        selected_domain_code: selectedDomainCode.value ?? undefined,
        selected_sector: selectedSector.value ?? undefined,
        selected_intention: selectedIntention.value ?? undefined,
      },
      {
        onToken(token) {
          const idx = messages.value.length - 1
          if (idx >= 0 && messages.value[idx]?.role === 'assistant') {
            const prev = messages.value[idx]!
            messages.value = [
              ...messages.value.slice(0, idx),
              { role: 'assistant' as const, content: prev.content + token },
            ]
            nextTick(() => scrollToBottom())
          }
        },
        onDone(payload) {
          // Stocker les cas suggérés et le contexte pour la prochaine requête (détail « le 2 », ok, domaine/secteur)
          lastSuggestedCases.value = payload.suggested_cases ?? null
          const previousDomain = selectedDomainCode.value
          const previousSector = selectedSector.value
          if (payload.selected_domain_code !== undefined) selectedDomainCode.value = payload.selected_domain_code
          if (payload.selected_sector !== undefined) selectedSector.value = payload.selected_sector
          if (payload.selected_intention !== undefined) selectedIntention.value = payload.selected_intention
          if (selectedDomainCode.value !== previousDomain) {
            selectedSector.value = null
            selectedIntention.value = null
          } else if (selectedSector.value !== previousSector) {
            selectedIntention.value = null
          }
          if (payload.pending_action !== undefined) pendingAction.value = payload.pending_action
          if (payload.pending_use_case_id !== undefined) pendingUseCaseId.value = payload.pending_use_case_id
          loading.value = false
          nextTick(() => {
            scrollToBottom()
            chatInputRef.value?.focus()
          })
        },
        onError(msg) {
          error.value = msg
          const idx = messages.value.length - 1
          if (idx >= 0 && messages.value[idx]?.role === 'assistant') {
            messages.value = [
              ...messages.value.slice(0, idx),
              { role: 'assistant' as const, content: `Désolé, une erreur s'est produite : ${msg}` },
            ]
          }
          loading.value = false
          nextTick(() => chatInputRef.value?.focus())
        },
      }
    )
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Erreur inconnue'
    const last = messages.value[messages.value.length - 1]
    if (last?.role === 'assistant') {
      last.content = `Désolé, une erreur s'est produite : ${error.value}`
      messages.value = [...messages.value]
    }
    loading.value = false
    nextTick(() => chatInputRef.value?.focus())
  }
}

</script>

<template>
  <div class="rag-page">
    <header class="rag-header">
      <div class="header-title-row">
        <img
          :src="microsoftLogo"
          alt="Microsoft"
          class="logo-microsoft"
        />
        <h1>Avoulia v2</h1>
      </div>
      <p class="subtitle">Posez vos questions sur vos documents indexés.</p>
    </header>

    <section class="chat-section">
      <h2>Chat</h2>
      <div class="chat-container">
        <div class="messages">
          <template v-if="!messages.length">
            <div class="empty-state">
              Posez une question sur vos documents indexés.
            </div>
          </template>
          <template v-else>
            <div
              v-for="(msg, i) in messages"
              :key="`${i}-${msg.role}`"
              :class="['message', msg.role]"
            >
              <span class="message-role">{{ msg.role === 'user' ? 'Vous' : 'Assistant' }}</span>
              <div class="message-content">{{ msg.content }}</div>
            </div>
          </template>
          <div ref="messagesEnd" />
        </div>

        <div v-if="error" class="error-banner">{{ error }}</div>

        <div v-if="loading" class="typing-line typing-above-input">Réflexion…</div>

        <form class="input-row" @submit.prevent="submit">
          <input
            ref="chatInputRef"
            v-model="input"
            type="text"
            placeholder="Votre question…"
            class="input"
            :disabled="loading"
          />
          <button type="submit" class="btn btn-primary" :disabled="loading || !input.trim()">
            Envoyer
          </button>
        </form>
        <p class="chat-disclaimer">
          Vous interagissez avec un assistant IA. Les réponses peuvent être inexactes
          ou incomplètes.
          Ne saisissez pas de données personnelles, confidentielles ou sensibles.
          Vérifiez et adaptez les résultats avant utilisation.
        </p>
      </div>
    </section>
  </div>
</template>

<style scoped>
.rag-page {
  max-width: 720px;
  margin: 0 auto;
  padding: 1.25rem 1rem;
  display: flex;
  flex-direction: column;
  gap: 1.75rem;
}

.header-title-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.35rem;
}

.logo-microsoft {
  height: 36px;
  width: auto;
  display: block;
}

.rag-header h1 {
  font-size: 1.6rem;
  font-weight: 600;
  margin: 0;
  color: var(--color-heading);
  letter-spacing: -0.02em;
}

.subtitle {
  font-size: 0.95rem;
  color: var(--color-text-muted);
  margin: 0 0 0.25rem 0;
}

.chat-section {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 1.25rem;
  background: var(--color-background-soft);
  box-shadow: var(--shadow-sm);
}

.chat-section h2 {
  font-size: 0.85rem;
  font-weight: 600;
  margin: 0 0 1rem 0;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.btn {
  padding: 0.55rem 1.1rem;
  border-radius: var(--radius-sm);
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid var(--color-border);
  background: var(--color-background);
  color: var(--color-text);
  transition: background 0.2s, border-color 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}

.btn-primary:hover:not(:disabled) {
  background: var(--accent-hover);
  border-color: var(--accent-hover);
}

.chat-container {
  display: flex;
  flex-direction: column;
  min-height: 320px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  max-height: 380px;
  padding: 0.25rem 0;
}

.empty-state {
  text-align: center;
  color: var(--color-text-muted);
  padding: 2rem 1.5rem;
  font-size: 0.95rem;
}

.message {
  margin-bottom: 1.25rem;
}

.message-role {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-muted);
}

.message.user .message-role { color: var(--accent); }
.message.assistant .message-role { color: var(--color-text-muted); }

.message-content {
  margin-top: 0.25rem;
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.95rem;
}

.message.user .message-content {
  background: var(--accent-mute);
  color: var(--color-heading);
  margin-right: 2rem;
  border: 1px solid rgba(9, 105, 218, 0.2);
}

.message.assistant .message-content {
  background: var(--color-background);
  border: 1px solid var(--color-border);
  margin-left: 2rem;
  color: var(--color-text);
}

.typing-line.typing-above-input {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  padding: 0.4rem 0 0.6rem 0;
  border-top: 1px solid var(--color-border);
  margin-top: 0.5rem;
}

@media (prefers-color-scheme: dark) {
  .message.user .message-content {
    background: rgba(9, 105, 218, 0.2);
    color: var(--vt-c-text-dark-1);
    border-color: rgba(33, 139, 255, 0.25);
  }
}

.error-banner {
  padding: 0.65rem 1rem;
  background: rgba(207, 34, 46, 0.1);
  color: #cf222e;
  font-size: 0.875rem;
  border-radius: var(--radius-sm);
  margin-top: 0.5rem;
  border: 1px solid rgba(207, 34, 46, 0.2);
}

.input-row {
  display: flex;
  gap: 0.6rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--color-border);
  margin-top: 0.5rem;
}

.input {
  flex: 1;
  padding: 0.7rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 1rem;
  font-family: inherit;
  background: var(--color-background);
  color: var(--color-text);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-mute);
}

.chat-disclaimer {
  margin: 0.75rem 0 0;
  font-size: 0.8rem;
  line-height: 1.45;
  color: var(--color-text-muted);
}
</style>
