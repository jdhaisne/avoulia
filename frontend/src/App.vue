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

@media (hover: hover) {
  .app-header a:hover {
    color: var(--accent);
  }
}
</style>
