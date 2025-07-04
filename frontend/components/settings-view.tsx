"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { User, Bell, Shield, Palette, Database, Mail, Smartphone, Save, Edit } from "lucide-react"

export function SettingsView() {
  const [isEditing, setIsEditing] = useState(false)
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    alerts: true,
    reports: true,
    anomalies: true,
  })

  const [profile, setProfile] = useState({
    name: "John Doe",
    email: "john.doe@company.com",
    role: "Data Analyst",
    department: "Business Intelligence",
    timezone: "UTC-8 (Pacific Time)",
  })

  const handleNotificationChange = (key: string, value: boolean) => {
    setNotifications((prev) => ({ ...prev, [key]: value }))
  }

  const handleSave = () => {
    setIsEditing(false)
    // In a real app, this would save to backend
    console.log("Settings saved")
  }

  return (
    <div className="flex flex-col h-full bg-gray-950 p-6 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white mb-2">Settings</h1>
          <p className="text-gray-400">Manage your account and preferences</p>
        </div>
        <Button
          onClick={() => (isEditing ? handleSave() : setIsEditing(true))}
          className="bg-blue-600 hover:bg-blue-700"
        >
          {isEditing ? (
            <>
              <Save className="w-4 h-4 mr-2" />
              Save Changes
            </>
          ) : (
            <>
              <Edit className="w-4 h-4 mr-2" />
              Edit Profile
            </>
          )}
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Profile Settings */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <User className="w-5 h-5" />
              <span>Profile Information</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center space-x-4 mb-4">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                <span className="text-xl font-bold text-white">JD</span>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">{profile.name}</h3>
                <p className="text-gray-400">{profile.role}</p>
                <Badge variant="secondary" className="bg-blue-600/20 text-blue-400 border-blue-600/30 mt-1">
                  {profile.department}
                </Badge>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <Label htmlFor="name" className="text-gray-300">
                  Full Name
                </Label>
                <Input
                  id="name"
                  value={profile.name}
                  onChange={(e) => setProfile((prev) => ({ ...prev, name: e.target.value }))}
                  disabled={!isEditing}
                  className="bg-gray-700 border-gray-600 text-white disabled:opacity-60"
                />
              </div>
              <div>
                <Label htmlFor="email" className="text-gray-300">
                  Email Address
                </Label>
                <Input
                  id="email"
                  type="email"
                  value={profile.email}
                  onChange={(e) => setProfile((prev) => ({ ...prev, email: e.target.value }))}
                  disabled={!isEditing}
                  className="bg-gray-700 border-gray-600 text-white disabled:opacity-60"
                />
              </div>
              <div>
                <Label htmlFor="role" className="text-gray-300">
                  Role
                </Label>
                <Input
                  id="role"
                  value={profile.role}
                  onChange={(e) => setProfile((prev) => ({ ...prev, role: e.target.value }))}
                  disabled={!isEditing}
                  className="bg-gray-700 border-gray-600 text-white disabled:opacity-60"
                />
              </div>
              <div>
                <Label htmlFor="department" className="text-gray-300">
                  Department
                </Label>
                <Input
                  id="department"
                  value={profile.department}
                  onChange={(e) => setProfile((prev) => ({ ...prev, department: e.target.value }))}
                  disabled={!isEditing}
                  className="bg-gray-700 border-gray-600 text-white disabled:opacity-60"
                />
              </div>
              <div>
                <Label htmlFor="timezone" className="text-gray-300">
                  Timezone
                </Label>
                <select
                  id="timezone"
                  value={profile.timezone}
                  onChange={(e) => setProfile((prev) => ({ ...prev, timezone: e.target.value }))}
                  disabled={!isEditing}
                  className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-3 py-2 disabled:opacity-60"
                >
                  <option value="UTC-8 (Pacific Time)">UTC-8 (Pacific Time)</option>
                  <option value="UTC-5 (Eastern Time)">UTC-5 (Eastern Time)</option>
                  <option value="UTC+0 (GMT)">UTC+0 (GMT)</option>
                  <option value="UTC+1 (CET)">UTC+1 (CET)</option>
                </select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Notification Settings */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Bell className="w-5 h-5" />
              <span>Notification Preferences</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Mail className="w-4 h-4 text-gray-400" />
                  <div>
                    <p className="text-white font-medium">Email Notifications</p>
                    <p className="text-gray-400 text-sm">Receive updates via email</p>
                  </div>
                </div>
                <Switch
                  checked={notifications.email}
                  onCheckedChange={(checked) => handleNotificationChange("email", checked)}
                />
              </div>

              <Separator className="bg-gray-700" />

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Smartphone className="w-4 h-4 text-gray-400" />
                  <div>
                    <p className="text-white font-medium">Push Notifications</p>
                    <p className="text-gray-400 text-sm">Browser push notifications</p>
                  </div>
                </div>
                <Switch
                  checked={notifications.push}
                  onCheckedChange={(checked) => handleNotificationChange("push", checked)}
                />
              </div>

              <Separator className="bg-gray-700" />

              <div className="space-y-3">
                <h4 className="text-white font-medium">Alert Types</h4>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-white">Critical Alerts</p>
                    <p className="text-gray-400 text-sm">High priority system alerts</p>
                  </div>
                  <Switch
                    checked={notifications.alerts}
                    onCheckedChange={(checked) => handleNotificationChange("alerts", checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-white">Report Generation</p>
                    <p className="text-gray-400 text-sm">When reports are completed</p>
                  </div>
                  <Switch
                    checked={notifications.reports}
                    onCheckedChange={(checked) => handleNotificationChange("reports", checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-white">Anomaly Detection</p>
                    <p className="text-gray-400 text-sm">Data anomalies and outliers</p>
                  </div>
                  <Switch
                    checked={notifications.anomalies}
                    onCheckedChange={(checked) => handleNotificationChange("anomalies", checked)}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Theme Settings */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Palette className="w-5 h-5" />
              <span>Appearance</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label className="text-gray-300 mb-3 block">Theme</Label>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-gray-900 border-2 border-blue-500 rounded-lg cursor-pointer">
                  <div className="w-full h-8 bg-gray-800 rounded mb-2"></div>
                  <div className="space-y-1">
                    <div className="w-3/4 h-2 bg-gray-700 rounded"></div>
                    <div className="w-1/2 h-2 bg-gray-700 rounded"></div>
                  </div>
                  <p className="text-white text-sm mt-2 text-center">Dark (Current)</p>
                </div>
                <div className="p-3 bg-white border-2 border-gray-600 rounded-lg cursor-pointer opacity-50">
                  <div className="w-full h-8 bg-gray-200 rounded mb-2"></div>
                  <div className="space-y-1">
                    <div className="w-3/4 h-2 bg-gray-300 rounded"></div>
                    <div className="w-1/2 h-2 bg-gray-300 rounded"></div>
                  </div>
                  <p className="text-gray-800 text-sm mt-2 text-center">Light (Coming Soon)</p>
                </div>
              </div>
            </div>

            <Separator className="bg-gray-700" />

            <div>
              <Label className="text-gray-300 mb-3 block">Accent Color</Label>
              <div className="flex space-x-2">
                <div className="w-8 h-8 bg-blue-500 rounded-full border-2 border-white cursor-pointer"></div>
                <div className="w-8 h-8 bg-teal-500 rounded-full border-2 border-gray-600 cursor-pointer"></div>
                <div className="w-8 h-8 bg-purple-500 rounded-full border-2 border-gray-600 cursor-pointer"></div>
                <div className="w-8 h-8 bg-green-500 rounded-full border-2 border-gray-600 cursor-pointer"></div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Security Settings */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Shield className="w-5 h-5" />
              <span>Security & Privacy</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              <div>
                <h4 className="text-white font-medium mb-2">Password</h4>
                <Button variant="outline" className="border-gray-600 text-gray-300 bg-transparent">
                  Change Password
                </Button>
              </div>

              <Separator className="bg-gray-700" />

              <div>
                <h4 className="text-white font-medium mb-2">Two-Factor Authentication</h4>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-300">Enable 2FA for additional security</p>
                    <Badge variant="secondary" className="bg-green-600/20 text-green-400 border-green-600/30 mt-1">
                      Enabled
                    </Badge>
                  </div>
                  <Button variant="outline" size="sm" className="border-gray-600 text-gray-300 bg-transparent">
                    Configure
                  </Button>
                </div>
              </div>

              <Separator className="bg-gray-700" />

              <div>
                <h4 className="text-white font-medium mb-2">Data Privacy</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Share usage analytics</span>
                    <Switch defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300">Allow data processing for insights</span>
                    <Switch defaultChecked />
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Data Management */}
        <Card className="bg-gray-800 border-gray-700 lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Database className="w-5 h-5" />
              <span>Data Management</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-gray-700 rounded-lg">
                <h4 className="text-2xl font-bold text-white mb-1">3</h4>
                <p className="text-gray-400">Active Datasets</p>
                <p className="text-xs text-gray-500 mt-1">4.2 MB total</p>
              </div>
              <div className="text-center p-4 bg-gray-700 rounded-lg">
                <h4 className="text-2xl font-bold text-white mb-1">12</h4>
                <p className="text-gray-400">Generated Reports</p>
                <p className="text-xs text-gray-500 mt-1">Last 30 days</p>
              </div>
              <div className="text-center p-4 bg-gray-700 rounded-lg">
                <h4 className="text-2xl font-bold text-white mb-1">847</h4>
                <p className="text-gray-400">Chat Messages</p>
                <p className="text-xs text-gray-500 mt-1">This month</p>
              </div>
            </div>

            <Separator className="bg-gray-700 my-6" />

            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-white font-medium">Data Retention</h4>
                <p className="text-gray-400 text-sm">Automatically delete old data after 12 months</p>
              </div>
              <Button variant="outline" className="border-gray-600 text-gray-300 bg-transparent">
                Configure
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
