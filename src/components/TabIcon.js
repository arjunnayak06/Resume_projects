import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const TabIcon = ({ name, focused, color, size }) => {
  const getIcon = () => {
    switch (name) {
      case 'home':
        return focused ? '🏠' : '🏠';
      case 'bar-chart':
        return focused ? '📊' : '📊';
      case 'user':
        return focused ? '👤' : '👤';
      case 'settings':
        return focused ? '⚙️' : '⚙️';
      default:
        return '🏠';
    }
  };

  return (
    <View style={styles.container}>
      <Text style={[styles.icon, { color, opacity: focused ? 1 : 0.6 }]}>
        {getIcon()}
      </Text>
      {focused && <View style={styles.indicator} />}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    fontSize: 24,
    marginBottom: 2,
  },
  indicator: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#6366f1',
    marginTop: 2,
  },
});

export default TabIcon;