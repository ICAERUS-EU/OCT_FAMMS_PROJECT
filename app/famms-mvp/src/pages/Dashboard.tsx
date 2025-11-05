import { useState, useMemo } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Camera, Download } from "lucide-react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import DetailedForestMap from "@/components/DetailedForestMap";
import ImageComparisonSlider from "@/components/ImageComparisonSlider";
import beforeImage from "@/assets/before-deforestation.png";
import afterImage from "@/assets/after-deforestation.png";

interface MarkerData {
  id: number;
  lat: number;
  lng: number;
  area: string;
  status: "alert" | "clear";
  date: string;
  confidence: number;
  beforeImage: string;
  afterImage: string;
}

const mockMarkers: MarkerData[] = [
  {
    id: 1,
    lat: 45.2,
    lng: 25.5,
    area: "Zona A-12",
    status: "alert",
    date: "2025-10-07",
    confidence: 87,
    beforeImage,
    afterImage,
  },
  {
    id: 2,
    lat: 45.3,
    lng: 25.7,
    area: "Zona B-08",
    status: "alert",
    date: "2025-10-06",
    confidence: 92,
    beforeImage,
    afterImage,
  },
  {
    id: 3,
    lat: 45.1,
    lng: 25.6,
    area: "Zona C-15",
    status: "clear",
    date: "2025-10-08",
    confidence: 15,
    beforeImage,
    afterImage,
  },
];

const Dashboard = () => {
  const [selectedMarker, setSelectedMarker] = useState<MarkerData | null>(null);
  const [dateRange, setDateRange] = useState("7days");
  const [riskLevel, setRiskLevel] = useState("all");
  const [detectionType, setDetectionType] = useState("all");
  const [area, setArea] = useState("all");
  const [droneSource, setDroneSource] = useState("all");

  // Filter markers based on selected filters
  const filteredMarkers = useMemo(() => {
    return mockMarkers.filter((marker) => {
      // Risk level filter based on confidence
      if (riskLevel !== "all") {
        if (riskLevel === "low" && marker.confidence >= 30) return false;
        if (riskLevel === "medium" && (marker.confidence < 30 || marker.confidence > 70)) return false;
        if (riskLevel === "high" && marker.confidence <= 70) return false;
      }

      // Area filter
      if (area !== "all") {
        const markerZone = marker.area.split("-")[0].toLowerCase().replace("zona ", "zone ");
        const selectedZone = `zone ${area}`;
        if (!markerZone.includes(selectedZone)) return false;
      }

      return true;
    });
  }, [riskLevel, area]);

  return (
    <DashboardLayout 
      title="Map View"
      dateRange={dateRange}
      riskLevel={riskLevel}
      detectionType={detectionType}
      area={area}
      droneSource={droneSource}
      onDateRangeChange={setDateRange}
      onRiskLevelChange={setRiskLevel}
      onDetectionTypeChange={setDetectionType}
      onAreaChange={setArea}
      onDroneSourceChange={setDroneSource}
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
        {/* Map Area */}
        <Card className="lg:col-span-2 shadow-md">
          <CardHeader>
            <CardTitle>Forest Monitoring Map</CardTitle>
          </CardHeader>
          <CardContent className="relative h-[600px] bg-muted rounded-lg overflow-hidden">
            <DetailedForestMap
              markers={filteredMarkers}
              onMarkerClick={setSelectedMarker}
            />
          </CardContent>
        </Card>

        {/* Stats Cards */}
        <div className="space-y-4">
          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-accent">
                {filteredMarkers.filter((m) => m.status === "alert").length}
              </div>
              <p className="text-xs text-muted-foreground mt-1">Requires attention</p>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Monitored Areas</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-primary">{filteredMarkers.length}</div>
              <p className="text-xs text-muted-foreground mt-1">Total zones tracked</p>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Last Scan</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-lg font-semibold">2 hours ago</div>
              <p className="text-xs text-muted-foreground mt-1">Zone A-12 completed</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Detail Sheet */}
      <Sheet open={!!selectedMarker} onOpenChange={() => setSelectedMarker(null)}>
        <SheetContent className="w-full sm:max-w-xl overflow-y-auto">
          {selectedMarker && (
            <>
              <SheetHeader>
                <SheetTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Drone Surveillance - {selectedMarker.area}
                </SheetTitle>
              </SheetHeader>

              <div className="mt-6 space-y-6">
                {/* Alert Badge */}
                {selectedMarker.status === "alert" && (
                  <div className="flex items-center gap-2 p-3 bg-accent/10 border border-accent rounded-lg">
                    <AlertTriangle className="h-5 w-5 text-accent" />
                    <div>
                      <p className="font-semibold text-accent">Possible Illegal Deforestation Detected</p>
                      <p className="text-sm text-muted-foreground">
                        Confidence: {selectedMarker.confidence}%
                      </p>
                    </div>
                  </div>
                )}

                {/* Metadata */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Area</p>
                    <p className="font-semibold">{selectedMarker.area}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Date</p>
                    <p className="font-semibold">{selectedMarker.date}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Status</p>
                    <Badge variant={selectedMarker.status === "alert" ? "destructive" : "default"}>
                      {selectedMarker.status === "alert" ? "alert" : "normal"}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Confidence</p>
                    <p className="font-semibold">{selectedMarker.confidence}%</p>
                  </div>
                  <div className="col-span-2">
                    <p className="text-sm text-muted-foreground">Coordinates</p>
                    <p className="font-semibold font-mono text-sm">
                      {selectedMarker.lat.toFixed(6)}°N, {selectedMarker.lng.toFixed(6)}°E
                    </p>
                  </div>
                </div>

                {/* Image Comparison */}
                <div className="space-y-2">
                  <p className="text-sm font-medium">Image Comparison</p>
                  <ImageComparisonSlider
                    beforeImage={selectedMarker.beforeImage}
                    afterImage={selectedMarker.afterImage}
                    beforeLabel="Before"
                    afterLabel="After"
                  />
                </div>

                {/* Actions */}
                <Button className="w-full bg-gradient-forest hover:opacity-90">
                  <Download className="h-4 w-4 mr-2" />
                  Generate Report
                </Button>
              </div>
            </>
          )}
        </SheetContent>
      </Sheet>
    </DashboardLayout>
  );
};

export default Dashboard;
