"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Database, Upload, Search, Calendar, CheckCircle, AlertCircle, Eye, Download, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface Dataset {
  id: string
  name: string
  uploadDate: Date
  size: string
  rows: number
  columns: number
  status: "active" | "processing" | "error"
  description: string
  fileType: string
}

interface DatasetsViewProps {
  onDatasetSelect: (datasetId: string) => void
  selectedDataset: string | null
}

const mockDatasets: Dataset[] = [
  {
    id: "sales-2023",
    name: "Sales Data 2023",
    uploadDate: new Date("2023-12-15"),
    size: "2.4 MB",
    rows: 15847,
    columns: 12,
    status: "active",
    description:
      "Complete sales transactions for 2023 including customer data, product information, and revenue metrics.",
    fileType: "CSV",
  },
  {
    id: "customer-analytics",
    name: "Customer Analytics",
    uploadDate: new Date("2023-12-10"),
    size: "1.8 MB",
    rows: 8932,
    columns: 18,
    status: "active",
    description: "Customer behavior analysis data with demographics, purchase history, and engagement metrics.",
    fileType: "XLSX",
  },
  {
    id: "inventory-data",
    name: "Inventory Management",
    uploadDate: new Date("2023-12-08"),
    size: "956 KB",
    rows: 3421,
    columns: 8,
    status: "processing",
    description: "Current inventory levels, stock movements, and supplier information.",
    fileType: "JSON",
  },
]

export function DatasetsView({ onDatasetSelect, selectedDataset }: DatasetsViewProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedDatasetDetails, setSelectedDatasetDetails] = useState<Dataset | null>(null)

  const filteredDatasets = mockDatasets.filter(
    (dataset) =>
      dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      dataset.description.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const getStatusIcon = (status: Dataset["status"]) => {
    switch (status) {
      case "active":
        return <CheckCircle className="w-4 h-4 text-green-400" />
      case "processing":
        return <AlertCircle className="w-4 h-4 text-yellow-400" />
      case "error":
        return <AlertCircle className="w-4 h-4 text-red-400" />
    }
  }

  const getStatusColor = (status: Dataset["status"]) => {
    switch (status) {
      case "active":
        return "bg-green-600/20 text-green-400 border-green-600/30"
      case "processing":
        return "bg-yellow-600/20 text-yellow-400 border-yellow-600/30"
      case "error":
        return "bg-red-600/20 text-red-400 border-red-600/30"
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-950 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Datasets</h1>
          <p className="text-gray-400">Manage and analyze your business data</p>
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700">
          <Upload className="w-4 h-4 mr-2" />
          Upload Dataset
        </Button>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
        <Input
          placeholder="Search datasets..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10 bg-gray-800 border-gray-600 text-white placeholder-gray-400"
        />
      </div>

      {/* Datasets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 flex-1 overflow-y-auto">
        {filteredDatasets.map((dataset) => (
          <Card
            key={dataset.id}
            className={cn(
              "bg-gray-800 border-gray-700 hover:border-gray-600 transition-colors cursor-pointer",
              selectedDataset === dataset.id && "border-blue-500 bg-blue-600/10",
            )}
            onClick={() => onDatasetSelect(dataset.id)}
          >
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2">
                  <Database className="w-5 h-5 text-blue-400" />
                  <CardTitle className="text-white text-lg">{dataset.name}</CardTitle>
                </div>
                <Badge variant="secondary" className={getStatusColor(dataset.status)}>
                  {getStatusIcon(dataset.status)}
                  <span className="ml-1 capitalize">{dataset.status}</span>
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-gray-300 text-sm line-clamp-2">{dataset.description}</p>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">Rows</p>
                  <p className="text-white font-medium">{dataset.rows.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-400">Columns</p>
                  <p className="text-white font-medium">{dataset.columns}</p>
                </div>
                <div>
                  <p className="text-gray-400">Size</p>
                  <p className="text-white font-medium">{dataset.size}</p>
                </div>
                <div>
                  <p className="text-gray-400">Type</p>
                  <p className="text-white font-medium">{dataset.fileType}</p>
                </div>
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                <div className="flex items-center text-xs text-gray-400">
                  <Calendar className="w-3 h-3 mr-1" />
                  {dataset.uploadDate.toLocaleDateString()}
                </div>
                <div className="flex items-center space-x-2">
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          setSelectedDatasetDetails(dataset)
                        }}
                        className="text-gray-400 hover:text-white"
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="bg-gray-800 border-gray-700 text-white max-w-2xl">
                      <DialogHeader>
                        <DialogTitle className="flex items-center space-x-2">
                          <Database className="w-5 h-5 text-blue-400" />
                          <span>{selectedDatasetDetails?.name}</span>
                        </DialogTitle>
                      </DialogHeader>
                      {selectedDatasetDetails && (
                        <div className="space-y-6">
                          <div>
                            <h4 className="text-sm font-medium text-gray-300 mb-2">Description</h4>
                            <p className="text-gray-100">{selectedDatasetDetails.description}</p>
                          </div>

                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <h4 className="text-sm font-medium text-gray-300 mb-2">Dataset Info</h4>
                              <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-gray-400">Rows:</span>
                                  <span className="text-white">{selectedDatasetDetails.rows.toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">Columns:</span>
                                  <span className="text-white">{selectedDatasetDetails.columns}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">Size:</span>
                                  <span className="text-white">{selectedDatasetDetails.size}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-gray-400">Type:</span>
                                  <span className="text-white">{selectedDatasetDetails.fileType}</span>
                                </div>
                              </div>
                            </div>

                            <div>
                              <h4 className="text-sm font-medium text-gray-300 mb-2">Status</h4>
                              <div className="space-y-2">
                                <Badge variant="secondary" className={getStatusColor(selectedDatasetDetails.status)}>
                                  {getStatusIcon(selectedDatasetDetails.status)}
                                  <span className="ml-1 capitalize">{selectedDatasetDetails.status}</span>
                                </Badge>
                                <p className="text-xs text-gray-400">
                                  Uploaded on {selectedDatasetDetails.uploadDate.toLocaleDateString()}
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
                              <Download className="w-4 h-4 mr-2" />
                              Download
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              className="border-red-600 text-red-400 hover:bg-red-600/20 bg-transparent"
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </Button>
                          </div>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredDatasets.length === 0 && (
        <div className="flex flex-col items-center justify-center flex-1 text-center">
          <Database className="w-16 h-16 text-gray-600 mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">No datasets found</h3>
          <p className="text-gray-500 mb-4">
            {searchTerm ? "Try adjusting your search terms" : "Upload your first dataset to get started"}
          </p>
          <Button className="bg-blue-600 hover:bg-blue-700">
            <Upload className="w-4 h-4 mr-2" />
            Upload Dataset
          </Button>
        </div>
      )}
    </div>
  )
}
