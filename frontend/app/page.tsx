"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import { Sidebar } from "@/components/sidebar"
import { ChatInterface } from "@/components/chat-interface"
import { DatasetsView } from "@/components/datasets-view"
import { ReportsView } from "@/components/reports-view"
import { AlertsView } from "@/components/alerts-view"
import { SettingsView } from "@/components/settings-view"
import { SidebarProvider } from "@/components/ui/sidebar"
import { Loader2 } from "lucide-react"

export type ViewType = "chat" | "datasets" | "reports" | "alerts" | "settings"

export default function Home() {
  const [currentView, setCurrentView] = useState<ViewType>("chat")
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login')
    }
  }, [isAuthenticated, isLoading, router])

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-950">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return null
  }

  const renderView = () => {
    switch (currentView) {
      case "chat":
        return <ChatInterface selectedDataset={selectedDataset} />
      case "datasets":
        return <DatasetsView onDatasetSelect={setSelectedDataset} selectedDataset={selectedDataset} />
      case "reports":
        return <ReportsView />
      case "alerts":
        return <AlertsView />
      case "settings":
        return <SettingsView />
      default:
        return <ChatInterface selectedDataset={selectedDataset} />
    }
  }

  return (
    <SidebarProvider>
      <div className="flex h-screen bg-gray-950">
        <Sidebar currentView={currentView} onViewChange={setCurrentView} />
        <main className="flex-1 overflow-hidden">{renderView()}</main>
      </div>
    </SidebarProvider>
  )
}
