# Matrix Health Terminal

A Matrix-style terminal interface for mental health counseling built with Next.js.

## Features

- 🧠 Mental health counseling interface
- 🔮 Matrix-inspired terminal UI with glowing effects
- 🔒 Secure password-protected input for sensitive information
- 🌐 Fully responsive for all devices
- 🚀 Easy to deploy to Vercel

## Getting Started

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. Clone this repository
```bash
git clone <repository-url>
cd matrix-health-frontend
```

2. Install dependencies
```bash
npm install
# or
yarn install
```

3. Configure environment variables
Create a `.env.local` file in the root directory with the following:
```
BACKEND_URL=http://localhost:8000
```

### Running the app locally

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Connect to Backend

This frontend is designed to connect to your Chainlit mental health counseling backend. Make sure your backend is running on http://localhost:8000 or update the `BACKEND_URL` environment variable.

## Deployment

### Deploy to Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new).

1. Push your code to a Git repository (GitHub, GitLab, or Bitbucket)
2. Import your project to Vercel
3. Configure environment variables in the Vercel dashboard
4. Deploy!

## Customization

- Edit `globals.css` to change the color scheme and animations
- Modify the responses in `page.tsx` to change the AI behavior
- Update the API route in `app/api/chat/route.ts` to connect to your backend

## License

MIT
