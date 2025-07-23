import { Suspense } from "react"
import SupportChat from "@/components/support-chat"
import Header from "@/components/header"

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-white">
      <Header />
      <main className="flex-1 container mx-auto px-4 py-8 md:py-16 max-w-7xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-serif font-medium tracking-tight mb-4">How can we help?</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Connect with our support team through chat or voice. We're here to assist you with any questions about your
            Aven card or home equity.
          </p>
        </div>

        <div className="max-w-3xl mx-auto">
          <Suspense fallback={<div className="h-[600px] w-full bg-gray-50 rounded-lg animate-pulse"></div>}>
            <SupportChat />
          </Suspense>
        </div>
      </main>
      <footer className="py-8 border-t border-gray-100">
        <div className="container mx-auto px-4 text-center text-sm text-gray-500">
          © {new Date().getFullYear()} Aven. All rights reserved.
        </div>
      </footer>
    </div>
  )
}
