# 📱 Mobile App Interface

A modern, cross-platform mobile application built with React Native and Expo, featuring a beautiful UI for data science and analytics workflows.

## ✨ Features

- **📊 Dashboard**: Overview of projects, statistics, and quick actions
- **📈 Analytics**: Data visualization with interactive charts and metrics
- **👤 Profile**: User profile with achievements, skills, and activity tracking
- **⚙️ Settings**: Comprehensive settings with toggles and preferences
- **🎨 Modern UI**: Beautiful gradient designs and smooth animations
- **📱 Cross-Platform**: Works on iOS, Android, and Web

## 🚀 Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)

### Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Run on different platforms:
   ```bash
   # iOS Simulator
   npm run ios
   
   # Android Emulator
   npm run android
   
   # Web Browser
   npm run web
   ```

## 📁 Project Structure

```
├── App.js                 # Main app component with navigation
├── package.json          # Dependencies and scripts
├── app.json              # Expo configuration
├── src/
│   ├── components/       # Reusable UI components
│   │   └── TabIcon.js   # Custom tab bar icons
│   └── screens/         # Main app screens
│       ├── HomeScreen.js     # Dashboard with stats and projects
│       ├── DataScreen.js     # Analytics and data visualization
│       ├── ProfileScreen.js  # User profile and achievements
│       └── SettingsScreen.js # App settings and preferences
└── README.md
```

## 🎨 Design System

### Colors
- **Primary**: `#6366f1` (Indigo)
- **Secondary**: `#8b5cf6` (Purple)
- **Background**: `#f8fafc` (Light Gray)
- **Text**: `#1f2937` (Dark Gray)
- **Accent**: Various gradients for visual appeal

### Components
- **Cards**: Rounded corners with subtle shadows
- **Buttons**: Gradient backgrounds with smooth transitions
- **Navigation**: Bottom tabs with custom icons
- **Charts**: Interactive data visualizations

## 📱 Screens Overview

### 🏠 Home Screen
- Welcome message and project overview
- Statistics grid with gradient cards
- Recent projects with progress indicators
- Quick action buttons

### 📊 Data Screen
- Performance analytics with interactive charts
- Key metrics with trend indicators
- ML model status and accuracy
- Data source management

### 👤 Profile Screen
- User information with gradient header
- Achievement badges and skills progress
- Recent activity timeline
- Profile management actions

### ⚙️ Settings Screen
- Organized settings sections
- Toggle switches for preferences
- App information and version details
- Sign out functionality

## 🛠 Technologies Used

- **React Native**: Cross-platform mobile development
- **Expo**: Development platform and build tools
- **React Navigation**: Navigation library
- **Linear Gradient**: Beautiful gradient effects
- **JavaScript ES6+**: Modern JavaScript features

## 📦 Dependencies

```json
{
  "expo": "~49.0.0",
  "react": "18.2.0",
  "react-native": "0.72.6",
  "@react-navigation/native": "^6.1.7",
  "@react-navigation/bottom-tabs": "^6.5.8",
  "expo-linear-gradient": "~12.3.0"
}
```

## 🚀 Deployment

### Expo Build
```bash
# Build for iOS
expo build:ios

# Build for Android
expo build:android

# Web deployment
expo build:web
```

### Expo Application Services (EAS)
```bash
# Install EAS CLI
npm install -g @expo/eas-cli

# Build with EAS
eas build --platform all
```

## 🎯 Future Enhancements

- [ ] Real-time data synchronization
- [ ] Push notifications
- [ ] Offline data caching
- [ ] Dark mode theme
- [ ] Advanced analytics
- [ ] User authentication
- [ ] Data export features

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

---

Built with ❤️ using React Native and Expo