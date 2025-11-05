import { useState } from "react";
import { DashboardLayout } from "@/components/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Plus, CheckCircle, Clock, AlertCircle } from "lucide-react";

interface Mission {
  id: string;
  area: string;
  date: string;
  status: "completed" | "in-progress" | "scheduled";
  coverage: number;
  droneId: string;
}

const mockMissions: Mission[] = [
  {
    id: "MSN-001",
    area: "Zone A-12",
    date: "2025-10-07",
    status: "completed",
    coverage: 100,
    droneId: "DRN-Alpha-01",
  },
  {
    id: "MSN-002",
    area: "Zone B-08",
    date: "2025-10-06",
    status: "completed",
    coverage: 100,
    droneId: "DRN-Beta-02",
  },
  {
    id: "MSN-003",
    area: "Zone D-22",
    date: "2025-10-09",
    status: "in-progress",
    coverage: 67,
    droneId: "DRN-Alpha-01",
  },
  {
    id: "MSN-004",
    area: "Zone F-30",
    date: "2025-10-10",
    status: "scheduled",
    coverage: 0,
    droneId: "DRN-Gamma-03",
  },
  {
    id: "MSN-005",
    area: "Zone C-15",
    date: "2025-10-08",
    status: "completed",
    coverage: 100,
    droneId: "DRN-Beta-02",
  },
];

const Missions = () => {
  const [showSuccessDialog, setShowSuccessDialog] = useState(false);

  const handleStartMission = () => {
    setShowSuccessDialog(true);
    setTimeout(() => setShowSuccessDialog(false), 3000);
  };

  const getStatusIcon = (status: Mission["status"]) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-primary" />;
      case "in-progress":
        return <Clock className="h-4 w-4 text-accent" />;
      case "scheduled":
        return <AlertCircle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: Mission["status"]) => {
    switch (status) {
      case "completed":
        return <Badge variant="default">Completed</Badge>;
      case "in-progress":
        return <Badge variant="default" className="bg-accent text-accent-foreground">In Progress</Badge>;
      case "scheduled":
        return <Badge variant="secondary">Scheduled</Badge>;
    }
  };

  return (
    <DashboardLayout title="Drone Missions">
      <div className="space-y-6">
        {/* Header Card */}
        <Card className="shadow-sm">
          <CardContent className="pt-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div>
                <h2 className="text-2xl font-bold">Mission Control</h2>
                <p className="text-muted-foreground mt-1">
                  Schedule and monitor drone surveillance missions
                </p>
              </div>
              <Button
                onClick={handleStartMission}
                className="bg-gradient-forest hover:opacity-90"
              >
                <Plus className="h-4 w-4 mr-2" />
                Start New Mission
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Total Missions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-primary">{mockMissions.length}</div>
              <p className="text-xs text-muted-foreground mt-1">All time</p>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Active Missions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-accent">
                {mockMissions.filter((m) => m.status === "in-progress").length}
              </div>
              <p className="text-xs text-muted-foreground mt-1">Currently flying</p>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-primary">98%</div>
              <p className="text-xs text-muted-foreground mt-1">Mission completion</p>
            </CardContent>
          </Card>
        </div>

        {/* Missions Table */}
        <Card className="shadow-md">
          <CardHeader>
            <CardTitle>Mission History</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Mission ID</TableHead>
                  <TableHead>Area</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Drone</TableHead>
                  <TableHead>Coverage</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {mockMissions.map((mission) => (
                  <TableRow key={mission.id}>
                    <TableCell className="font-medium">{mission.id}</TableCell>
                    <TableCell>{mission.area}</TableCell>
                    <TableCell>{mission.date}</TableCell>
                    <TableCell className="text-muted-foreground text-sm">
                      {mission.droneId}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="w-16 bg-muted rounded-full h-2 overflow-hidden">
                          <div
                            className="h-full bg-primary transition-all"
                            style={{ width: `${mission.coverage}%` }}
                          />
                        </div>
                        <span className="text-sm">{mission.coverage}%</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {getStatusIcon(mission.status)}
                        {getStatusBadge(mission.status)}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {/* Success Dialog */}
      <Dialog open={showSuccessDialog} onOpenChange={setShowSuccessDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <div className="flex justify-center mb-4">
              <div className="bg-primary/10 p-3 rounded-full">
                <CheckCircle className="h-12 w-12 text-primary" />
              </div>
            </div>
            <DialogTitle className="text-center">Mission Started Successfully!</DialogTitle>
            <DialogDescription className="text-center">
              Drone DRN-Alpha-01 is now en route to the designated area. You can monitor the
              mission progress in real-time.
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </DashboardLayout>
  );
};

export default Missions;
