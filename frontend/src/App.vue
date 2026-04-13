<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="app">
    <header class="app-header">
      <nav>
        <RouterLink to="/">Chat RAG</RouterLink>
      </nav>
    </header>
    <main class="app-main">
      <RouterView />
    </main>
    <footer class="app-footer">
      <p>
        Ce chatbot est opéré par Simplon, avec le soutien technologique de
        Microsoft. Il fournit des idées pour explorer des cas d’usage de l’IA
        générative. Il ne constitue pas un conseil professionnel.
      </p>
      <div class="footer-links">
        <RouterLink to="/confidentialite" class="footer-link">NOTICE DE CONFIDENTIALITE</RouterLink>
        <RouterLink to="/faq" class="footer-link">FAQ</RouterLink>
        <RouterLink to="/conditions-utilisation" class="footer-link">CONDITIONS D’UTILISATION</RouterLink>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  padding: 0.875rem 1.5rem;
  background: var(--color-background-soft);
  border-bottom: 1px solid var(--color-border);
}

.app-header nav {
  display: flex;
  gap: 1.25rem;
  align-items: center;
}

.app-header a {
  color: var(--color-text-muted);
  text-decoration: none;
  font-weight: 500;
  font-size: 0.95rem;
  transition: color 0.2s;
}

.app-header a.router-link-active {
  color: var(--accent);
}

.app-main {
  flex: 1;
}

.app-footer {
  padding: 0.9rem 1.5rem 1.1rem;
  border-top: 1px solid var(--color-border);
  background: var(--color-background-soft);
  color: var(--color-text-muted);
  font-size: 0.8rem;
  line-height: 1.45;
}

.app-footer p {
  margin: 0 0 0.35rem;
}

.footer-link {
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
}

.footer-links {
  display: flex;
  gap: 0.9rem;
  flex-wrap: wrap;
}

.footer-link:hover {
  text-decoration: underline;
}

@media (hover: hover) {
  .app-header a:hover {
    color: var(--accent);
  }
}
</style>
