/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0a1628",
        blueprint: "#0d1b2e",
        panel: "#0f1d32",
        card: "#111e36",
        border: "#1e3a5f",
        line: "#2d4a6f",
        accent: {
          DEFAULT: "#06b6d4", // Surveillance Cyan
          light: "#22d3ee",
          dark: "#0891b2",
        },
      },
      boxShadow: {
        glow: "0 0 12px 2px rgba(6, 182, 212, 0.15), 0 0 4px 1px rgba(6, 182, 212, 0.1)",
        card: "0 4px 12px -2px rgba(0, 0, 0, 0.4), 0 2px 6px -1px rgba(0, 0, 0, 0.3)",
        inset: "inset 0 1px 2px rgba(0, 0, 0, 0.2)",
      },
    },
  },
  plugins: [],
};
