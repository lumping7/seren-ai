import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(input: string | number | Date): string {
  const date = new Date(input)
  return date.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  })
}

export function getInitials(name: string) {
  if (!name) return "??"
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .substring(0, 2)
}

export function truncateText(text: string, maxLength: number) {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + "..."
}

export function getRandomGradient() {
  const gradients = [
    "from-purple-500 to-indigo-500",
    "from-blue-500 to-purple-500",
    "from-indigo-500 to-blue-500",
    "from-violet-500 to-purple-500",
    "from-fuchsia-500 to-purple-500",
  ]
  return gradients[Math.floor(Math.random() * gradients.length)]
}

export function getDomainFromUrl(url: string) {
  try {
    const domain = new URL(url).hostname
    return domain
  } catch (e) {
    return url
  }
}

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))