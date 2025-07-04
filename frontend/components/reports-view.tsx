"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { FileText, Search, Calendar, TrendingUp, BarChart3, PieChart, Download, Eye, Share, Filter } from "lucide-react"

interface Report {
  id: string
  title: string
  description: string
  createdDate: Date
  type: "sales" | "customer" | "financial" | "operational"
  status: "completed" | "generating" | "scheduled"
  insights: number
  charts: number
  dataset: string
}

const mockReports: Report[] = [
  {
    id: "sales-q4-2023",
    title: "Q4 2023 Sales Performance",
    description:
      "Comprehensive analysis of Q4 sales data including revenue trends, top products, and regional performance.",
    createdDate: new Date("2023-12-20"),
    type: "sales",
    status: "completed",
    insights: 12,
    charts: 8,
    dataset: "Sales Data 2023",
  },
  {
    id: "customer-segmentation",
    title: "Customer Segmentation Analysis",
    description: "Deep dive into customer behavior patterns, lifetime value analysis, and segmentation strategies.",
    createdDate: new Date("2023-12-18"),
    type: "customer",
    status: "completed",
    insights: 15,
    charts: 6,
    dataset: "Customer Analytics",
  },
  {
    id: "inventory-optimization",
    title: "Inventory Optimization Report",
    description: "Analysis of inventory levels, turnover rates, and recommendations for stock optimization.",
    createdDate: new Date("2023-12-15"),
    type: "operational",
    status: "generating",
    insights: 8,
    charts: 5,
    dataset: "Inventory Management",
  },
  {
    id: "financial-summary",
    title: "Monthly Financial Summary",
    description: "Complete financial overview including P&L analysis, cash flow, and budget variance.",
    createdDate: new Date("2023-12-12"),
    type: "financial",
    status: "completed",
    insights: 10,
    charts: 12,
    dataset: "Financial Data",
  },
]

export function ReportsView() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedReport, setSelectedReport] = useState<Report | null>(null)
  const [filterType, setFilterType] = useState<string>("all")

  const filteredReports = mockReports.filter((report) => {
    const matchesSearch =
      report.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filterType === "all" || report.type === filterType
    return matchesSearch && matchesFilter
  })

  const getTypeIcon = (type: Report["type"]) => {
    switch (type) {
      case "sales":
        return <TrendingUp className="w-4 h-4" />
      case "customer":
        return <BarChart3 className="w-4 h-4" />
      case "financial":
        return <PieChart className="w-4 h-4" />
      case "operational":
        return <FileText className="w-4 h-4" />
    }
  }

  const getTypeColor = (type: Report["type"]) => {
    switch (type) {
      case "sales":
        return "bg-green-600/20 text-green-400 border-green-600/30"
      case "customer":
        return "bg-blue-600/20 text-blue-400 border-blue-600/30"
      case "financial":
        return "bg-purple-600/20 text-purple-400 border-purple-600/30"
      case "operational":
        return "bg-orange-600/20 text-orange-400 border-orange-600/30"
    }
  }

  const getStatusColor = (status: Report["status"]) => {
    switch (status) {
      case "completed":
        return "bg-green-600/20 text-green-400 border-green-600/30"
      case "generating":
        return "bg-yellow-600/20 text-yellow-400 border-yellow-600/30"
      case "scheduled":
        return "bg-blue-600/20 text-blue-400 border-blue-600/30"
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-950 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Reports</h1>
          <p className="text-gray-400">Generated insights and analytics reports</p>
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700">
          <FileText className="w-4 h-4 mr-2" />
          Generate Report
        </Button>
      </div>

      {/* Search and Filter */}
      <div className="flex items-center space-x-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search reports..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-gray-800 border-gray-600 text-white placeholder-gray-400"
          />
        </div>
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="bg-gray-800 border border-gray-600 text-white rounded-md px-3 py-2 text-sm"
          >
            <option value="all">All Types</option>
            <option value="sales">Sales</option>
            <option value="customer">Customer</option>
            <option value="financial">Financial</option>
            <option value="operational">Operational</option>
          </select>
        </div>
      </div>

      {/* Reports Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 flex-1 overflow-y-auto">
        {filteredReports.map((report) => (
          <Card key={report.id} className="bg-gray-800 border-gray-700 hover:border-gray-600 transition-colors">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2">
                  {getTypeIcon(report.type)}
                  <CardTitle className="text-white text-lg">{report.title}</CardTitle>
                </div>
                <Badge variant="secondary" className={getStatusColor(report.status)}>
                  {report.status}
                </Badge>
              </div>
              <Badge variant="secondary" className={getTypeColor(report.type)}>
                {report.type}
              </Badge>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-gray-300 text-sm line-clamp-2">{report.description}</p>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Insights</p>
                  <p className="text-white font-medium">{report.insights}</p>
                </div>
                <div>
                  <p className="text-gray-400">Charts</p>
                  <p className="text-white font-medium">{report.charts}</p>
                </div>
              </div>

              <div className="text-sm">
                <p className="text-gray-400">Dataset</p>
                <p className="text-blue-400">{report.dataset}</p>
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                <div className="flex items-center text-xs text-gray-400">
                  <Calendar className="w-3 h-3 mr-1" />
                  {report.createdDate.toLocaleDateString()}
                </div>
                <div className="flex items-center space-x-2">
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedReport(report)}
                        className="text-gray-400 hover:text-white"
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="bg-gray-800 border-gray-700 text-white max-w-4xl max-h-[80vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle className="flex items-center space-x-2">
                          {selectedReport && getTypeIcon(selectedReport.type)}
                          <span>{selectedReport?.title}</span>
                        </DialogTitle>
                      </DialogHeader>
                      {selectedReport && (
                        <div className="space-y-6">
                          <div className="flex items-center space-x-4">
                            <Badge variant="secondary" className={getTypeColor(selectedReport.type)}>
                              {selectedReport.type}
                            </Badge>
                            <Badge variant="secondary" className={getStatusColor(selectedReport.status)}>
                              {selectedReport.status}
                            </Badge>
                          </div>

                          <div>
                            <h4 className="text-sm font-medium text-gray-300 mb-2">Description</h4>
                            <p className="text-gray-100">{selectedReport.description}</p>
                          </div>

                          <div className="grid grid-cols-3 gap-4">
                            <div className="text-center p-4 bg-gray-700 rounded-lg">
                              <p className="text-2xl font-bold text-white">{selectedReport.insights}</p>
                              <p className="text-sm text-gray-400">Key Insights</p>
                            </div>
                            <div className="text-center p-4 bg-gray-700 rounded-lg">
                              <p className="text-2xl font-bold text-white">{selectedReport.charts}</p>
                              <p className="text-sm text-gray-400">Visualizations</p>
                            </div>
                            <div className="text-center p-4 bg-gray-700 rounded-lg">
                              <p className="text-2xl font-bold text-white">98%</p>
                              <p className="text-sm text-gray-400">Accuracy</p>
                            </div>
                          </div>

                          <div>
                            <h4 className="text-sm font-medium text-gray-300 mb-4">Key Findings</h4>
                            <div className="space-y-3">
                              <div className="p-3 bg-gray-700 rounded-lg">
                                <h5 className="font-medium text-white mb-1">Revenue Growth</h5>
                                <p className="text-sm text-gray-300">
                                  23% increase in quarterly revenue compared to previous period
                                </p>
                              </div>
                              <div className="p-3 bg-gray-700 rounded-lg">
                                <h5 className="font-medium text-white mb-1">Customer Acquisition</h5>
                                <p className="text-sm text-gray-300">
                                  Customer acquisition cost decreased by 12% while retention improved
                                </p>
                              </div>
                              <div className="p-3 bg-gray-700 rounded-lg">
                                <h5 className="font-medium text-white mb-1">Market Performance</h5>
                                <p className="text-sm text-gray-300">
                                  Electronics category showing strongest performance with 45% of total sales
                                </p>
                              </div>
                            </div>
                          </div>

                          <div className="flex justify-end space-x-2 pt-4 border-t border-gray-700">
                            <Button
                              variant="outline"
                              size="sm"
                              className="border-gray-600 text-gray-300 bg-transparent"
                            >
                              <Share className="w-4 h-4 mr-2" />
                              Share
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              className="border-gray-600 text-gray-300 bg-transparent"
                            >
                              <Download className="w-4 h-4 mr-2" />
                              Export
                            </Button>
                          </div>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                  <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredReports.length === 0 && (
        <div className="flex flex-col items-center justify-center flex-1 text-center">
          <FileText className="w-16 h-16 text-gray-600 mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No reports found</h3>
          <p className="text-gray-500 mb-4">
            {searchTerm ? "Try adjusting your search terms" : "Generate your first report to get started"}
          </p>
          <Button className="bg-blue-600 hover:bg-blue-700">
            <FileText className="w-4 h-4 mr-2" />
            Generate Report
          </Button>
        </div>
      )}
    </div>
  )
}
