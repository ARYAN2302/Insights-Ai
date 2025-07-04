"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  AlertTriangle,
  Search,
  Calendar,
  TrendingUp,
  TrendingDown,
  Bell,
  Eye,
  X,
  CheckCircle,
  Clock,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface Alert {
  id: string
  title: string
  description: string
  type: "anomaly" | "trend" | "threshold" | "opportunity"
  severity: "low" | "medium" | "high" | "critical"
  createdDate: Date
  dataset: string
  status: "new" | "acknowledged" | "resolved"
  metric: string
  value: string
  change: number
}

const mockAlerts: Alert[] = [
  {
    id: "alert-1",
    title: "Unusual Sales Spike Detected",
    description:
      "Sales volume increased by 340% in the Electronics category over the past 24 hours. This may indicate a successful promotion or data anomaly.",
    type: "anomaly",
    severity: "high",
    createdDate: new Date("2023-12-20T14:30:00"),
    dataset: "Sales Data 2023",
    status: "new",
    metric: "Daily Sales Volume",
    value: "$127,450",
    change: 340,
  },
  {
    id: "alert-2",
    title: "Customer Churn Rate Increasing",
    description:
      "Customer churn rate has increased by 15% over the past week. Immediate attention recommended to prevent further losses.",
    type: "trend",
    severity: "critical",
    createdDate: new Date("2023-12-20T10:15:00"),
    dataset: "Customer Analytics",
    status: "acknowledged",
    metric: "Churn Rate",
    value: "12.3%",
    change: 15,
  },
  {
    id: "alert-3",
    title: "Inventory Level Below Threshold",
    description: "Multiple products are running low on inventory. Reorder recommended to avoid stockouts.",
    type: "threshold",
    severity: "medium",
    createdDate: new Date("2023-12-19T16:45:00"),
    dataset: "Inventory Management",
    status: "new",
    metric: "Stock Level",
    value: "23 items",
    change: -45,
  },
  {
    id: "alert-4",
    title: "New Market Opportunity Identified",
    description:
      "Analysis shows potential for 25% revenue increase in the Asia Pacific region based on current trends.",
    type: "opportunity",
    severity: "low",
    createdDate: new Date("2023-12-19T09:20:00"),
    dataset: "Sales Data 2023",
    status: "new",
    metric: "Revenue Potential",
    value: "+$2.1M",
    change: 25,
  },
  {
    id: "alert-5",
    title: "Payment Processing Delays",
    description:
      "Average payment processing time has increased by 200% in the last 48 hours. System investigation required.",
    type: "anomaly",
    severity: "high",
    createdDate: new Date("2023-12-18T13:10:00"),
    dataset: "Financial Data",
    status: "resolved",
    metric: "Processing Time",
    value: "4.2 minutes",
    change: 200,
  },
]

export function AlertsView() {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterSeverity, setFilterSeverity] = useState<string>("all")
  const [filterStatus, setFilterStatus] = useState<string>("all")

  const filteredAlerts = mockAlerts.filter((alert) => {
    const matchesSearch =
      alert.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      alert.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesSeverity = filterSeverity === "all" || alert.severity === filterSeverity
    const matchesStatus = filterStatus === "all" || alert.status === filterStatus
    return matchesSearch && matchesSeverity && matchesStatus
  })

  const getTypeIcon = (type: Alert["type"]) => {
    switch (type) {
      case "anomaly":
        return <AlertTriangle className="w-4 h-4" />
      case "trend":
        return <TrendingUp className="w-4 h-4" />
      case "threshold":
        return <Bell className="w-4 h-4" />
      case "opportunity":
        return <TrendingUp className="w-4 h-4" />
    }
  }

  const getSeverityColor = (severity: Alert["severity"]) => {
    switch (severity) {
      case "low":
        return "bg-blue-600/20 text-blue-400 border-blue-600/30"
      case "medium":
        return "bg-yellow-600/20 text-yellow-400 border-yellow-600/30"
      case "high":
        return "bg-orange-600/20 text-orange-400 border-orange-600/30"
      case "critical":
        return "bg-red-600/20 text-red-400 border-red-600/30"
    }
  }

  const getStatusIcon = (status: Alert["status"]) => {
    switch (status) {
      case "new":
        return <Clock className="w-3 h-3" />
      case "acknowledged":
        return <Eye className="w-3 h-3" />
      case "resolved":
        return <CheckCircle className="w-3 h-3" />
    }
  }

  const getStatusColor = (status: Alert["status"]) => {
    switch (status) {
      case "new":
        return "bg-blue-600/20 text-blue-400 border-blue-600/30"
      case "acknowledged":
        return "bg-yellow-600/20 text-yellow-400 border-yellow-600/30"
      case "resolved":
        return "bg-green-600/20 text-green-400 border-green-600/30"
    }
  }

  const handleStatusChange = (alertId: string, newStatus: Alert["status"]) => {
    // In a real app, this would update the alert status
    console.log(`Changing alert ${alertId} status to ${newStatus}`)
  }

  return (
    <div className="flex flex-col h-full bg-gray-950 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Alerts</h1>
          <p className="text-gray-400">Proactive insights and anomaly detection</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="secondary" className="bg-red-600/20 text-red-400 border-red-600/30">
            {filteredAlerts.filter((a) => a.status === "new").length} New
          </Badge>
          <Button variant="outline" className="border-gray-600 text-gray-300 bg-transparent">
            <Bell className="w-4 h-4 mr-2" />
            Configure Alerts
          </Button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex items-center space-x-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search alerts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-gray-800 border-gray-600 text-white placeholder-gray-400"
          />
        </div>
        <select
          value={filterSeverity}
          onChange={(e) => setFilterSeverity(e.target.value)}
          className="bg-gray-800 border border-gray-600 text-white rounded-md px-3 py-2 text-sm"
        >
          <option value="all">All Severities</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
          <option value="critical">Critical</option>
        </select>
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="bg-gray-800 border border-gray-600 text-white rounded-md px-3 py-2 text-sm"
        >
          <option value="all">All Status</option>
          <option value="new">New</option>
          <option value="acknowledged">Acknowledged</option>
          <option value="resolved">Resolved</option>
        </select>
      </div>

      {/* Alerts List */}
      <div className="space-y-4 flex-1 overflow-y-auto">
        {filteredAlerts.map((alert) => (
          <Card
            key={alert.id}
            className={cn(
              "bg-gray-800 border-gray-700 hover:border-gray-600 transition-colors",
              alert.status === "new" && "border-l-4 border-l-blue-500",
            )}
          >
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  <div
                    className={cn(
                      "p-2 rounded-lg",
                      alert.severity === "critical" && "bg-red-600/20",
                      alert.severity === "high" && "bg-orange-600/20",
                      alert.severity === "medium" && "bg-yellow-600/20",
                      alert.severity === "low" && "bg-blue-600/20",
                    )}
                  >
                    {getTypeIcon(alert.type)}
                  </div>
                  <div className="flex-1">
                    <CardTitle className="text-white text-lg mb-1">{alert.title}</CardTitle>
                    <p className="text-gray-300 text-sm">{alert.description}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant="secondary" className={getSeverityColor(alert.severity)}>
                    {alert.severity}
                  </Badge>
                  <Badge variant="secondary" className={getStatusColor(alert.status)}>
                    {getStatusIcon(alert.status)}
                    <span className="ml-1 capitalize">{alert.status}</span>
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-gray-400 text-xs">Metric</p>
                  <p className="text-white font-medium">{alert.metric}</p>
                </div>
                <div>
                  <p className="text-gray-400 text-xs">Current Value</p>
                  <p className="text-white font-medium">{alert.value}</p>
                </div>
                <div>
                  <p className="text-gray-400 text-xs">Change</p>
                  <div className="flex items-center space-x-1">
                    {alert.change > 0 ? (
                      <TrendingUp className="w-3 h-3 text-green-400" />
                    ) : (
                      <TrendingDown className="w-3 h-3 text-red-400" />
                    )}
                    <span className={cn("font-medium", alert.change > 0 ? "text-green-400" : "text-red-400")}>
                      {alert.change > 0 ? "+" : ""}
                      {alert.change}%
                    </span>
                  </div>
                </div>
                <div>
                  <p className="text-gray-400 text-xs">Dataset</p>
                  <p className="text-blue-400 text-sm">{alert.dataset}</p>
                </div>
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-gray-700">
                <div className="flex items-center text-xs text-gray-400">
                  <Calendar className="w-3 h-3 mr-1" />
                  {alert.createdDate.toLocaleString()}
                </div>
                <div className="flex items-center space-x-2">
                  {alert.status === "new" && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleStatusChange(alert.id, "acknowledged")}
                      className="border-yellow-600 text-yellow-400 hover:bg-yellow-600/20"
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      Acknowledge
                    </Button>
                  )}
                  {alert.status === "acknowledged" && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleStatusChange(alert.id, "resolved")}
                      className="border-green-600 text-green-400 hover:bg-green-600/20"
                    >
                      <CheckCircle className="w-3 h-3 mr-1" />
                      Resolve
                    </Button>
                  )}
                  <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredAlerts.length === 0 && (
        <div className="flex flex-col items-center justify-center flex-1 text-center">
          <AlertTriangle className="w-16 h-16 text-gray-600 mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No alerts found</h3>
          <p className="text-gray-500 mb-4">
            {searchTerm
              ? "Try adjusting your search terms"
              : "All alerts are resolved or no alerts have been generated"}
          </p>
          <Button variant="outline" className="border-gray-600 text-gray-300 bg-transparent">
            <Bell className="w-4 h-4 mr-2" />
            Configure Alert Rules
          </Button>
        </div>
      )}
    </div>
  )
}
