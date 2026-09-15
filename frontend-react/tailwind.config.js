/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#070A0F', // deep void
          900: '#0B0F17', // main canvas
          850: '#111827', // panel surface
          800: '#1F2937', // border / card
          700: '#374151', // muted border
          600: '#4B5563', // secondary text
        },
        petro: {
          cyan: '#06b6d4',
          emerald: '#10b981',
          amber: '#f59e0b',
          blue: '#3b82f6',
          purple: '#8b5cf6',
          red: '#ef4444'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
