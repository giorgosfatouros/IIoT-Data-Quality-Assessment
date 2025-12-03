# IIoT Data Quality Frontend

React 18+ frontend application for visualizing and analyzing IIoT sensor data quality using TypeScript, Vite, and Tailwind CSS.

## Quick Start (Development)

For local development, the frontend runs separately from Docker:

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at http://localhost:5173

## Production Build

Build the production bundle:

```bash
cd frontend
npm install
npm run build
```

The optimized build output will be in the `dist/` directory.

### Preview Production Build

Test the production build locally:

```bash
npm run preview
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t fame-frontend ./frontend
```

### Run Container

```bash
docker run -p 5173:5173 fame-frontend
```

The application will be available at http://localhost:5173

### Docker Compose

For production deployment, uncomment the frontend service in `docker-compose.yml`:

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
  container_name: fame-frontend
  ports:
    - "5173:5173"
  environment:
    - VITE_API_URL=http://backend:8000
  depends_on:
    - backend
  networks:
    - fame-network
  restart: unless-stopped
```

Then start with:

```bash
docker-compose up -d frontend
```

## Environment Configuration

**Note**: Currently, the API base URL is hardcoded to `http://localhost:8000` in the source code. For production deployment, consider:

1. Using Vite environment variables (`VITE_API_URL`)
2. Updating all API calls to use `import.meta.env.VITE_API_URL`
3. Setting the environment variable in Docker Compose or deployment configuration

Example:

```typescript
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
```

## Frontend Structure

```
frontend/
├── src/
│   ├── components/      # Reusable React components
│   │   ├── Card.tsx
│   │   ├── Footer.tsx
│   │   └── SensorSelector.tsx
│   ├── pages/           # Page components (routes)
│   │   ├── Home.tsx
│   │   ├── DataLoading.tsx
│   │   ├── DataVisualization.tsx
│   │   ├── DataQuality.tsx
│   │   ├── MissingValues.tsx
│   │   ├── InvalidValues.tsx
│   │   └── DQAAgent.tsx
│   ├── assets/          # Static assets (images, icons)
│   ├── App.tsx          # Main app component with routing
│   ├── main.tsx         # Application entry point
│   └── styles.css       # Global styles
├── index.html           # HTML template
├── package.json         # Dependencies
├── vite.config.ts       # Vite configuration
├── tsconfig.json        # TypeScript configuration
├── tailwind.config.js   # Tailwind CSS configuration
└── Dockerfile           # Production Docker image
```

## Available Scripts

- `npm run dev` - Start development server with hot-reload
- `npm run build` - Build production bundle
- `npm run preview` - Preview production build locally

## Technology Stack

- **React 18+** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Recharts** - Data visualization
- **Lucide React** - Icon library

## API Integration

The frontend communicates with the backend API at `http://localhost:8000` (development) or the configured `VITE_API_URL` (production).

### Main API Endpoints Used

- `GET /tables` - List machine groups
- `GET /data` - Get aggregated insights data
- `GET /data/preprocessed` - Get preprocessed sensor data
- `GET /tags` - Get sensor metadata/thresholds
- `GET /analytics/*` - Various analytics endpoints
- `POST /import/upload` - Upload and import data files
- `POST /agent/chat/stream` - DQA Agent chat (streaming)

## Troubleshooting

### Build Issues
- **Module not found**: Run `npm install` to install dependencies
- **Type errors**: Check `tsconfig.json` configuration
- **Build fails**: Ensure Node.js version is 18+ (check with `node --version`)

### Development Server
- **Port already in use**: Change port in `vite.config.ts` or kill existing process
- **Hot-reload not working**: Restart dev server
- **API connection errors**: Verify backend is running on port 8000

### Production Deployment
- **API calls fail**: Ensure `VITE_API_URL` is set correctly for production
- **Static assets not loading**: Check base path in `vite.config.ts`
- **CORS errors**: Configure CORS in backend to allow frontend origin

## Development Notes

- The frontend uses React Router for client-side routing
- All API calls use native `fetch` API
- State management is handled with React hooks (useState, useEffect)
- Tailwind CSS classes are used for styling
- TypeScript interfaces define data structures for API responses

