"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Calendar } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

type MeetingConfirmation = {
  meeting_id: string
  scheduled_date: string
  scheduled_time: string
  message: string
}

export default function MeetingScheduler() {
  const [email, setEmail] = useState("")
  const [topic, setTopic] = useState("")
  const [preferredDate, setPreferredDate] = useState("")
  const [preferredTime, setPreferredTime] = useState("")
  const [additionalNotes, setAdditionalNotes] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmation, setConfirmation] = useState<MeetingConfirmation | null>(null)

  // Get today's date in YYYY-MM-DD format for min date attribute
  const today = new Date().toISOString().split('T')[0]

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setError(null)
    setIsSubmitting(true)

    try {
      const response = await fetch(`${API_BASE_URL}/schedule-meeting`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({
          email,
          topic,
          preferred_date: preferredDate,
          preferred_time: preferredTime,
          additional_notes: additionalNotes || undefined,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || "Failed to schedule meeting")
      }

      setConfirmation(data)
      // Reset form
      setEmail("")
      setTopic("")
      setPreferredDate("")
      setPreferredTime("")
      setAdditionalNotes("")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Card className="w-full max-w-lg mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calendar className="h-5 w-5" />
          Schedule a Meeting
        </CardTitle>
        <CardDescription>
          Schedule a meeting with an Aven specialist to discuss your specific needs.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {confirmation ? (
          <div className="space-y-4">
            <Alert className="bg-green-50 border-green-200">
              <AlertTitle className="text-green-800">Meeting Scheduled!</AlertTitle>
              <AlertDescription className="text-green-700">
                {confirmation.message}
              </AlertDescription>
            </Alert>
            <div className="bg-gray-50 p-4 rounded-md space-y-2">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Meeting ID:</span>
                <span className="text-sm">{confirmation.meeting_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Date:</span>
                <span className="text-sm">{confirmation.scheduled_date}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Time:</span>
                <span className="text-sm">{confirmation.scheduled_time}</span>
              </div>
            </div>
            <p className="text-sm text-gray-600">
              A confirmation email has been sent to your email address with all the details.
            </p>
            <Button 
              onClick={() => setConfirmation(null)} 
              variant="outline" 
              className="w-full"
            >
              Schedule Another Meeting
            </Button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <Input
                id="email"
                type="email"
                placeholder="your@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="topic">Meeting Topic</Label>
              <Input
                id="topic"
                placeholder="What would you like to discuss?"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                required
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="preferredDate">Preferred Date</Label>
                <Input
                  id="preferredDate"
                  type="date"
                  min={today}
                  value={preferredDate}
                  onChange={(e) => setPreferredDate(e.target.value)}
                  required
                />
                <p className="text-xs text-gray-500">Business days only</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="preferredTime">Preferred Time</Label>
                <Input
                  id="preferredTime"
                  type="time"
                  min="09:00"
                  max="17:00"
                  value={preferredTime}
                  onChange={(e) => setPreferredTime(e.target.value)}
                  required
                />
                <p className="text-xs text-gray-500">9:00 AM - 5:00 PM</p>
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="additionalNotes">Additional Notes (Optional)</Label>
              <Textarea
                id="additionalNotes"
                placeholder="Any additional information that might help us prepare for the meeting"
                value={additionalNotes}
                onChange={(e) => setAdditionalNotes(e.target.value)}
                rows={3}
              />
            </div>
            
            <Button 
              type="submit"
              className="w-full" 
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  <span>Scheduling...</span>
                </div>
              ) : (
                "Schedule Meeting"
              )}
            </Button>
          </form>
        )}
      </CardContent>
    </Card>
  )
} 