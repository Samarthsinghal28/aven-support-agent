import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function Header() {
  return (
    <header className="border-b border-gray-100">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center">
          <span className="text-2xl font-bold">Aven</span>
        </Link>

        <nav className="hidden md:flex items-center space-x-8">
          <Link href="#" className="text-sm font-medium hover:text-gray-600">
            Card
          </Link>
          <Link href="#" className="text-sm font-medium hover:text-gray-600">
            How It Works
          </Link>
          <Link href="#" className="text-sm font-medium hover:text-gray-600">
            Reviews
          </Link>
          <Link href="#" className="text-sm font-medium hover:text-gray-600">
            Support
          </Link>
          <Link href="#" className="text-sm font-medium hover:text-gray-600">
            App
          </Link>
          <Link href="#" className="text-sm font-medium hover:text-gray-600">
            Who We Are
          </Link>
        </nav>

        <Button variant="outline" className="hidden md:inline-flex bg-transparent">
          Sign In
        </Button>

        <button className="md:hidden">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="lucide lucide-menu"
          >
            <line x1="4" x2="20" y1="12" y2="12" />
            <line x1="4" x2="20" y1="6" y2="6" />
            <line x1="4" x2="20" y1="18" y2="18" />
          </svg>
        </button>
      </div>
    </header>
  )
}
