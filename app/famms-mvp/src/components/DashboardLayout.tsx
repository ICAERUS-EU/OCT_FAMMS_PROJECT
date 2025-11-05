import { ReactNode } from "react";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "./AppSidebar";
import { TreePine } from "lucide-react";

interface DashboardLayoutProps {
  children: ReactNode;
  title: string;
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

export function DashboardLayout({ 
  children, 
  title,
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
}: DashboardLayoutProps) {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background">
        <AppSidebar 
          dateRange={dateRange}
          riskLevel={riskLevel}
          detectionType={detectionType}
          area={area}
          droneSource={droneSource}
          onDateRangeChange={onDateRangeChange}
          onRiskLevelChange={onRiskLevelChange}
          onDetectionTypeChange={onDetectionTypeChange}
          onAreaChange={onAreaChange}
          onDroneSourceChange={onDroneSourceChange}
        />
        <div className="flex-1 flex flex-col">
          <header className="h-16 border-b bg-card flex items-center px-6 gap-4 shadow-sm">
            <SidebarTrigger />
            <div className="flex items-center gap-3">
              <TreePine className="h-6 w-6 text-primary" />
              <h1 className="text-xl font-semibold text-foreground">{title}</h1>
            </div>
          </header>
          <main className="flex-1 p-6">{children}</main>
        </div>
      </div>
    </SidebarProvider>
  );
}
