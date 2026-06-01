# 🚚 Smart Chain Supply

An AI-powered logistics platform that predicts supply chain disruptions and dynamically optimizes shipment routes in real time.

## ✨ Features

- 📦 Real-Time Shipment Tracking
- 🤖 AI-Based Delay Prediction
- 🗺 Dynamic Route Optimization
- 🚨 Smart Disruption Alerts
- 📊 Analytics & KPI Dashboard
- 📡 Live Updates via WebSockets
- 🌦 Weather & Traffic Risk Detection
- 🏭 Warehouse Bottleneck Monitoring
- ⏱ ETA Prediction System
- 🔄 Automated Rerouting Recommendations

## 🛠 Tech Stack

**Frontend**
- React
- Tailwind CSS
- Mapbox
- Recharts

**Backend**
- FastAPI
- WebSockets
- SQLAlchemy

**Database**
- PostgreSQL
- TimescaleDB
- Redis

**AI / ML**
- Scikit-Learn
- XGBoost
- LightGBM

**Optimization**
- NetworkX
- Google OR-Tools

**DevOps**
- Docker
- Docker Compose
- Kafka

## 🏗 Architecture

```text
React Dashboard
       │
       ▼
    FastAPI
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
DB    ML   Optimizer
 │           │
 └────Kafka──┘
```

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/Smart-Chain-Supply.git

cd Smart-Chain-Supply

docker compose up --build
```

## 👥 Team

| Member | Role |
|----------|---------|
| Person A | Project Setup, Database, Synthetic Data, Route Graph |
| Person B | ML Models & Predictions |
| Person C | Backend APIs & Optimization |
| Person D | Frontend Dashboard & Maps |

## 🎯 Goal

Build a resilient supply chain system that proactively detects disruptions, predicts delays, and recommends optimized routes using AI and real-time analytics.

---

⭐ Star the repository if you like the project!
