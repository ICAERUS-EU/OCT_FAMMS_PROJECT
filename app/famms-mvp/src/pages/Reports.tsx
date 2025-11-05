import { useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Eye, AlertTriangle, Filter, ChevronDown } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface Report {
  id: string;
  area: string;
  date: string;
  confidence: number;
  status: "critical" | "warning" | "normal";
  description: string;
}

const mockReports: Report[] = [
  {
    id: "RPT-2025-001",
    area: "Zone A-12",
    date: "2025-10-07",
    confidence: 87,
    status: "critical",
    description: "Significant tree cover loss detected in protected area",
  },
  {
    id: "RPT-2025-002",
    area: "Zone B-08",
    date: "2025-10-06",
    confidence: 92,
    status: "critical",
    description: "Multiple deforestation sites identified through aerial surveillance",
  },
  {
    id: "RPT-2025-003",
    area: "Zone D-22",
    date: "2025-10-05",
    confidence: 68,
    status: "warning",
    description: "Unusual vehicle tracks observed near forest boundary",
  },
  {
    id: "RPT-2025-004",
    area: "Zone C-15",
    date: "2025-10-04",
    confidence: 15,
    status: "normal",
    description: "No suspicious activity detected during routine scan",
  },
  {
    id: "RPT-2025-005",
    area: "Zone E-19",
    date: "2025-10-03",
    confidence: 78,
    status: "warning",
    description: "Minor vegetation change detected, monitoring continues",
  },
];

const Reports = () => {
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [confidenceLevel, setConfidenceLevel] = useState([70]);

  const getStatusColor = (status: Report["status"]) => {
    switch (status) {
      case "critical":
        return "destructive";
      case "warning":
        return "default";
      default:
        return "secondary";
    }
  };

  return (
    <DashboardLayout title="Reports">
      <div className="space-y-6">
        {/* Header */}
        <Card className="shadow-sm">
          <CardContent className="pt-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div>
                <h2 className="text-2xl font-bold">Surveillance Reports</h2>
                <p className="text-muted-foreground mt-1">
                  Automated detection reports from drone monitoring missions
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Filters Panel */}
        <Collapsible open={filtersOpen} onOpenChange={setFiltersOpen}>
          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Filter className="h-4 w-4" />
                  <CardTitle className="text-base">Filters</CardTitle>
                </div>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <ChevronDown className={`h-4 w-4 transition-transform ${filtersOpen ? 'rotate-180' : ''}`} />
                  </Button>
                </CollapsibleTrigger>
              </div>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Status</label>
                    <Select defaultValue="all">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="pending">Pending</SelectItem>
                        <SelectItem value="validated">Validated</SelectItem>
                        <SelectItem value="flagged">Flagged</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Date Range</label>
                    <Select defaultValue="7days">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="7days">Last 7 days</SelectItem>
                        <SelectItem value="30days">Last month</SelectItem>
                        <SelectItem value="custom">Custom</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Export Type</label>
                    <Select defaultValue="pdf">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pdf">PDF</SelectItem>
                        <SelectItem value="csv">CSV</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      AI Confidence: ≥{confidenceLevel}%
                    </label>
                    <Slider
                      value={confidenceLevel}
                      onValueChange={setConfidenceLevel}
                      min={0}
                      max={100}
                      step={5}
                      className="mt-2"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>0%</span>
                      <span>100%</span>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <Button size="sm" className="bg-gradient-forest hover:opacity-90">
                    <Download className="h-4 w-4 mr-2" />
                    Export Filtered
                  </Button>
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>

        {/* Reports List */}
        <div className="space-y-4">
          {mockReports.map((report) => (
            <Card key={report.id} className="shadow-sm hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                  <div className="flex-1 space-y-3">
                    <div className="flex items-start gap-3">
                      {report.status === "critical" && (
                        <AlertTriangle className="h-5 w-5 text-accent mt-1 flex-shrink-0" />
                      )}
                      <div className="flex-1">
                        <div className="flex items-center gap-3 flex-wrap">
                          <h3 className="font-semibold text-lg">{report.id}</h3>
                          <Badge variant={getStatusColor(report.status)}>
                            {report.status.toUpperCase()}
                          </Badge>
                        </div>
                        <p className="text-muted-foreground mt-1">{report.description}</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Area</p>
                        <p className="font-medium">{report.area}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Date</p>
                        <p className="font-medium">{report.date}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Confidence</p>
                        <p className="font-medium">{report.confidence}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Detection Method</p>
                        <p className="font-medium">AI Analysis</p>
                      </div>
                    </div>
                  </div>

                  <div className="flex lg:flex-col gap-2">
                    <Button variant="outline" size="sm" className="flex-1 lg:flex-none">
                      <Eye className="h-4 w-4 mr-2" />
                      View
                    </Button>
                    <Button size="sm" className="flex-1 lg:flex-none">
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Reports;
