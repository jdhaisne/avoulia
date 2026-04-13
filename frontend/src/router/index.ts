import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import PrivacyView from '../views/PrivacyView.vue'
import FaqView from '../views/FaqView.vue'
import TermsView from '../views/TermsView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: ChatView,
    },
    {
      path: '/confidentialite',
      name: 'confidentialite',
      component: PrivacyView,
    },
    {
      path: '/faq',
      name: 'faq',
      component: FaqView,
    },
    {
      path: '/conditions-utilisation',
      name: 'conditions-utilisation',
      component: TermsView,
    },
  ],
})

export default router
