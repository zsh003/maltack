import { ref, computed } from 'vue'
import { theme } from 'ant-design-vue'

export default function useTheme() {
  const isDarkMode = ref(localStorage.getItem('theme') === 'dark')

  const themeMode = computed(() => (isDarkMode.value ? 'dark-theme' : 'light-theme'))
  const menuTheme = computed(() => (isDarkMode.value ? 'dark' : 'light'))
  const currentTheme = computed(() => ({
    algorithm: isDarkMode.value ? theme.darkAlgorithm : theme.defaultAlgorithm,
  }))

  const toggleTheme = () => {
    isDarkMode.value = !isDarkMode.value
    localStorage.setItem('theme', isDarkMode.value ? 'dark' : 'light')
  }

  return {
    isDarkMode,
    themeMode,
    menuTheme,
    currentTheme,
    toggleTheme,
  }
}
