import { MapPin, AlertTriangle } from "lucide-react";

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

interface DetailedForestMapProps {
  markers: MarkerData[];
  onMarkerClick: (marker: MarkerData) => void;
}

const DetailedForestMap = ({ markers, onMarkerClick }: DetailedForestMapProps) => {
  // Convert lat/lng to SVG coordinates
  const convertToSvgCoords = (lat: number, lng: number) => {
    const x = ((lng - 25.4) * 400 + 400);
    const y = ((45.25 - lat) * 400 + 300);
    return { x, y };
  };

  return (
    <div className="relative w-full h-full">
      <svg className="w-full h-full" viewBox="0 0 800 600" preserveAspectRatio="xMidYMid meet">
        <defs>
          {/* Forest pattern */}
          <pattern id="forestPattern" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="5" cy="5" r="2" fill="hsl(var(--primary))" opacity="0.3" />
            <circle cx="15" cy="5" r="2" fill="hsl(var(--primary))" opacity="0.3" />
            <circle cx="10" cy="15" r="2" fill="hsl(var(--primary))" opacity="0.3" />
          </pattern>
          
          {/* Mountain gradient */}
          <linearGradient id="mountainGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="hsl(var(--muted-foreground))" stopOpacity="0.6" />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0.3" />
          </linearGradient>

          {/* Water gradient */}
          <linearGradient id="waterGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.2" />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0.4" />
          </linearGradient>
        </defs>

        {/* Background - Forest base */}
        <rect width="800" height="600" fill="url(#forestPattern)" />
        
        {/* Rivers */}
        <path
          d="M 100,150 Q 200,180 300,150 T 500,180 T 700,150"
          stroke="url(#waterGradient)"
          strokeWidth="8"
          fill="none"
          opacity="0.6"
        />
        <path
          d="M 50,400 Q 150,420 250,400 T 450,420 T 750,400"
          stroke="url(#waterGradient)"
          strokeWidth="6"
          fill="none"
          opacity="0.5"
        />

        {/* Mountain ranges */}
        <g opacity="0.7">
          {/* Carpathian Mountains - North */}
          <path
            d="M 50,80 L 120,20 L 180,60 L 240,30 L 300,70 L 350,40 L 400,80 L 450,50 L 500,90 L 550,60 L 600,100 L 650,70 L 700,110 L 750,80 L 800,120"
            fill="url(#mountainGradient)"
            stroke="hsl(var(--muted-foreground))"
            strokeWidth="1"
            opacity="0.5"
          />
          
          {/* Southern Carpathians */}
          <path
            d="M 0,500 L 80,440 L 150,470 L 220,430 L 290,480 L 360,450 L 430,490 L 500,460 L 570,500 L 640,470 L 710,510 L 800,480"
            fill="url(#mountainGradient)"
            stroke="hsl(var(--muted-foreground))"
            strokeWidth="1"
            opacity="0.4"
          />
        </g>

        {/* Forest zones with different densities */}
        <circle cx="200" cy="250" r="80" fill="hsl(var(--primary))" opacity="0.15" />
        <circle cx="450" cy="300" r="100" fill="hsl(var(--primary))" opacity="0.2" />
        <circle cx="600" cy="200" r="70" fill="hsl(var(--primary))" opacity="0.15" />
        <circle cx="350" cy="450" r="90" fill="hsl(var(--primary))" opacity="0.18" />

        {/* Protected areas borders */}
        <rect x="150" y="200" width="180" height="150" 
          fill="none" 
          stroke="hsl(var(--primary))" 
          strokeWidth="2" 
          strokeDasharray="5,5" 
          opacity="0.4" 
        />
        <rect x="380" y="240" width="200" height="180" 
          fill="none" 
          stroke="hsl(var(--primary))" 
          strokeWidth="2" 
          strokeDasharray="5,5" 
          opacity="0.4" 
        />
        <rect x="250" y="380" width="160" height="140" 
          fill="none" 
          stroke="hsl(var(--primary))" 
          strokeWidth="2" 
          strokeDasharray="5,5" 
          opacity="0.4" 
        />

        {/* Location labels */}
        <g className="fill-foreground" style={{ fontSize: '12px', fontWeight: '600' }}>
          <text x="200" y="180" textAnchor="middle" opacity="0.7">Monti Carpazi</text>
          <text x="400" y="50" textAnchor="middle" opacity="0.7">Cresta Nord</text>
          <text x="650" y="520" textAnchor="middle" opacity="0.7">Valle del Sud</text>
          <text x="100" y="300" textAnchor="middle" opacity="0.7">Foresta Occidentale</text>
          <text x="650" y="250" textAnchor="middle" opacity="0.7">Riserva Orientale</text>
        </g>

        {/* Zone labels */}
        <g className="fill-muted-foreground" style={{ fontSize: '10px', fontWeight: '500' }}>
          <text x="240" y="270" textAnchor="middle">Zona A</text>
          <text x="480" y="330" textAnchor="middle">Zona B</text>
          <text x="330" y="450" textAnchor="middle">Zona C</text>
        </g>

        {/* Roads/trails */}
        <path
          d="M 0,300 L 200,310 L 400,295 L 600,305 L 800,300"
          stroke="hsl(var(--muted-foreground))"
          strokeWidth="2"
          strokeDasharray="10,5"
          fill="none"
          opacity="0.3"
        />

        {/* Grid reference lines */}
        <g stroke="hsl(var(--muted-foreground))" strokeWidth="0.5" opacity="0.1">
          <line x1="200" y1="0" x2="200" y2="600" />
          <line x1="400" y1="0" x2="400" y2="600" />
          <line x1="600" y1="0" x2="600" y2="600" />
          <line x1="0" y1="200" x2="800" y2="200" />
          <line x1="0" y1="400" x2="800" y2="400" />
        </g>
      </svg>

      {/* Interactive markers overlay */}
      {markers.map((marker) => {
        const { x, y } = convertToSvgCoords(marker.lat, marker.lng);
        return (
          <button
            key={marker.id}
            onClick={() => onMarkerClick(marker)}
            className="absolute transform -translate-x-1/2 -translate-y-1/2 transition-all hover:scale-125 z-10"
            style={{ left: `${(x / 800) * 100}%`, top: `${(y / 600) * 100}%` }}
          >
            <div className="relative">
              <MapPin
                className={`h-8 w-8 drop-shadow-lg ${
                  marker.status === "alert"
                    ? "text-accent fill-accent/20"
                    : "text-primary fill-primary/20"
                }`}
              />
              {marker.status === "alert" && (
                <AlertTriangle className="absolute -top-1 -right-1 h-4 w-4 text-destructive animate-pulse" />
              )}
            </div>
          </button>
        );
      })}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-card p-3 rounded-lg shadow-md space-y-2 z-10">
        <div className="flex items-center gap-2 text-sm">
          <MapPin className="h-4 w-4 text-accent" />
          <span>Allerta Rilevata</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <MapPin className="h-4 w-4 text-primary" />
          <span>Stato Normale</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground mt-2 pt-2 border-t">
          <div className="w-3 h-3 border-2 border-primary border-dashed" />
          <span>Area Protetta</span>
        </div>
      </div>

      {/* Compass */}
      <div className="absolute top-4 right-4 bg-card p-2 rounded-lg shadow-md z-10">
        <svg width="40" height="40" viewBox="0 0 40 40">
          <circle cx="20" cy="20" r="18" fill="none" stroke="hsl(var(--border))" strokeWidth="1" />
          <path d="M 20,5 L 23,20 L 20,18 L 17,20 Z" fill="hsl(var(--destructive))" />
          <text x="20" y="8" textAnchor="middle" className="text-xs font-bold fill-destructive">N</text>
          <text x="20" y="35" textAnchor="middle" className="text-xs fill-muted-foreground">S</text>
          <text x="8" y="23" textAnchor="middle" className="text-xs fill-muted-foreground">O</text>
          <text x="32" y="23" textAnchor="middle" className="text-xs fill-muted-foreground">E</text>
        </svg>
      </div>
    </div>
  );
};

export default DetailedForestMap;
