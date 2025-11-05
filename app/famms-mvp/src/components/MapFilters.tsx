import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SidebarGroup, SidebarGroupContent, SidebarGroupLabel } from "@/components/ui/sidebar";

interface MapFiltersProps {
  dateRange: string;
  riskLevel: string;
  detectionType: string;
  area: string;
  droneSource: string;
  onDateRangeChange: (value: string) => void;
  onRiskLevelChange: (value: string) => void;
  onDetectionTypeChange: (value: string) => void;
  onAreaChange: (value: string) => void;
  onDroneSourceChange: (value: string) => void;
}

export function MapFilters({
  dateRange,
  riskLevel,
  detectionType,
  area,
  droneSource,
  onDateRangeChange,
  onRiskLevelChange,
  onDetectionTypeChange,
  onAreaChange,
  onDroneSourceChange,
}: MapFiltersProps) {
  return (
    <SidebarGroup>
      <SidebarGroupLabel>Filters</SidebarGroupLabel>
      <SidebarGroupContent className="space-y-4 px-2">
        <div className="space-y-2">
          <label className="text-xs font-medium text-sidebar-foreground">Date Range</label>
          <Select value={dateRange} onValueChange={onDateRangeChange}>
            <SelectTrigger className="h-9 bg-sidebar-accent/50">
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
          <label className="text-xs font-medium text-sidebar-foreground">Risk Level</label>
          <Select value={riskLevel} onValueChange={onRiskLevelChange}>
            <SelectTrigger className="h-9 bg-sidebar-accent/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="low">Low (&lt;30%)</SelectItem>
              <SelectItem value="medium">Medium (30-70%)</SelectItem>
              <SelectItem value="high">High (&gt;70%)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-sidebar-foreground">Detection Type</label>
          <Select value={detectionType} onValueChange={onDetectionTypeChange}>
            <SelectTrigger className="h-9 bg-sidebar-accent/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="logging">Illegal logging</SelectItem>
              <SelectItem value="fire">Fire damage</SelectItem>
              <SelectItem value="clearing">Land clearing</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-sidebar-foreground">Area</label>
          <Select value={area} onValueChange={onAreaChange}>
            <SelectTrigger className="h-9 bg-sidebar-accent/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Zones</SelectItem>
              <SelectItem value="a">Zone A</SelectItem>
              <SelectItem value="b">Zone B</SelectItem>
              <SelectItem value="c">Zone C</SelectItem>
              <SelectItem value="d">Zone D</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-sidebar-foreground">Drone Source</label>
          <Select value={droneSource} onValueChange={onDroneSourceChange}>
            <SelectTrigger className="h-9 bg-sidebar-accent/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Drones</SelectItem>
              <SelectItem value="a">Drone A</SelectItem>
              <SelectItem value="b">Drone B</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </SidebarGroupContent>
    </SidebarGroup>
  );
}
