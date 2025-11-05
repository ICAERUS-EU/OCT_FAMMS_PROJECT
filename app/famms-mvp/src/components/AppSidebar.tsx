import { NavLink, useLocation } from "react-router-dom";
import { Map, FileText, Plane, TreePine } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";
import { MapFilters } from "./MapFilters";

interface AppSidebarProps {
  dateRange?: string;
  riskLevel?: string;
  detectionType?: string;
  area?: string;
  droneSource?: string;
  onDateRangeChange?: (value: string) => void;
  onRiskLevelChange?: (value: string) => void;
  onDetectionTypeChange?: (value: string) => void;
  onAreaChange?: (value: string) => void;
  onDroneSourceChange?: (value: string) => void;
}

const items = [
  { title: "Map View", url: "/dashboard", icon: Map },
  { title: "Reports", url: "/reports", icon: FileText },
  { title: "Drone Missions", url: "/missions", icon: Plane },
];

export function AppSidebar({
  dateRange = "7days",
  riskLevel = "all",
  detectionType = "all",
  area = "all",
  droneSource = "all",
  onDateRangeChange,
  onRiskLevelChange,
  onDetectionTypeChange,
  onAreaChange,
  onDroneSourceChange,
}: AppSidebarProps) {
  const location = useLocation();
  const isMapView = location.pathname === "/dashboard";

  return (
    <Sidebar>
      <SidebarHeader className="border-b border-sidebar-border p-4">
        <div className="flex items-center gap-3">
          <div className="bg-sidebar-accent p-2 rounded-lg">
            <TreePine className="h-6 w-6 text-sidebar-foreground" />
          </div>
          <div>
            <h2 className="font-bold text-sidebar-foreground">FAMMS</h2>
            <p className="text-xs text-sidebar-foreground/70">Forest Monitor</p>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      end={item.url === "/dashboard"}
                      className={({ isActive }) =>
                        isActive
                          ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                          : "hover:bg-sidebar-accent/50"
                      }
                    >
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {isMapView && onDateRangeChange && (
          <MapFilters
            dateRange={dateRange}
            riskLevel={riskLevel}
            detectionType={detectionType}
            area={area}
            droneSource={droneSource}
            onDateRangeChange={onDateRangeChange}
            onRiskLevelChange={onRiskLevelChange!}
            onDetectionTypeChange={onDetectionTypeChange!}
            onAreaChange={onAreaChange!}
            onDroneSourceChange={onDroneSourceChange!}
          />
        )}
      </SidebarContent>
    </Sidebar>
  );
}
