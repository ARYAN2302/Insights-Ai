"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { MessageSquare, Database, FileText, AlertTriangle, Settings, Menu, X, Brain } from "lucide-react"
import type { ViewType } from "@/app/page"
import { cn } from "@/lib/utils"

interface SidebarProps {
  currentView: ViewType
  onViewChange: (view: ViewType) => void
}

export function Sidebar({ currentView, onViewChange }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  const menuItems = [
    { id: "chat" as ViewType, label: "Chat", icon: MessageSquare, badge: null },
    { id: "datasets" as ViewType, label: "Datasets", icon: Database, badge: "3" },
    { id: "reports" as ViewType, label: "Reports", icon: FileText, badge: "12" },
    { id: "alerts" as ViewType, label: "Alerts", icon: AlertTriangle, badge: "5" },
    { id: "settings" as ViewType, label: "Settings", icon: Settings, badge: null },
  ]

  return (
    <>
      {/* Mobile overlay */}
      {!isCollapsed && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setIsCollapsed(true)} />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          "fixed lg:relative inset-y-0 left-0 z-50 w-64 bg-gray-900 border-r border-gray-800 transition-transform duration-300 ease-in-out",
          isCollapsed && "-translate-x-full lg:translate-x-0 lg:w-16",
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-800">
            <div className={cn("flex items-center space-x-2", isCollapsed && "lg:justify-center")}>
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              {!isCollapsed && (
                <div>
                  <h1 className="text-sm font-semibold text-white">GenAI Insights</h1>
                  <p className="text-xs text-gray-400">Business Suite</p>
                </div>
              )}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="lg:hidden text-gray-400 hover:text-white"
            >
              {isCollapsed ? <Menu className="w-4 h-4" /> : <X className="w-4 h-4" />}
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon
              const isActive = currentView === item.id

              return (
                <Button
                  key={item.id}
                  variant={isActive ? "secondary" : "ghost"}
                  className={cn(
                    "w-full justify-start text-left",
                    isActive
                      ? "bg-blue-600/20 text-blue-400 border border-blue-600/30"
                      : "text-gray-300 hover:text-white hover:bg-gray-800",
                    isCollapsed && "lg:justify-center lg:px-2",
                  )}
                  onClick={() => {
                    onViewChange(item.id)
                    setIsCollapsed(true) // Close on mobile after selection
                  }}
                >
                  <Icon className={cn("w-4 h-4", !isCollapsed && "mr-3")} />
                  {!isCollapsed && (
                    <>
                      <span className="flex-1">{item.label}</span>
                      {item.badge && (
                        <Badge variant="secondary" className="ml-auto bg-gray-700 text-gray-300">
                          {item.badge}
                        </Badge>
                      )}
                    </>
                  )}
                </Button>
              )
            })}
          </nav>

          {/* Footer */}
          {!isCollapsed && (
            <div className="p-4 border-t border-gray-800">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                  <span className="text-xs font-semibold text-white">JD</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">John Doe</p>
                  <p className="text-xs text-gray-400 truncate">Data Analyst</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Mobile toggle button */}
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsCollapsed(false)}
        className="fixed top-4 left-4 z-40 lg:hidden text-gray-400 hover:text-white bg-gray-900/80 backdrop-blur-sm"
      >
        <Menu className="w-4 h-4" />
      </Button>
    </>
  )
}
