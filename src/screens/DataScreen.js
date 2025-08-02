import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

const DataScreen = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('Week');
  
  const periods = ['Day', 'Week', 'Month', 'Year'];
  
  const chartData = [
    { label: 'Mon', value: 65, color: '#6366f1' },
    { label: 'Tue', value: 45, color: '#8b5cf6' },
    { label: 'Wed', value: 80, color: '#06b6d4' },
    { label: 'Thu', value: 35, color: '#10b981' },
    { label: 'Fri', value: 95, color: '#f59e0b' },
    { label: 'Sat', value: 25, color: '#ef4444' },
    { label: 'Sun', value: 55, color: '#ec4899' },
  ];

  const metrics = [
    { title: 'Accuracy', value: '94.2%', change: '+2.1%', positive: true },
    { title: 'Processing Time', value: '1.2s', change: '-0.3s', positive: true },
    { title: 'Error Rate', value: '2.1%', change: '-0.5%', positive: true },
    { title: 'Throughput', value: '850/h', change: '+15%', positive: true },
  ];

  const models = [
    { name: 'Linear Regression', accuracy: 92, status: 'Active' },
    { name: 'Random Forest', accuracy: 88, status: 'Training' },
    { name: 'Neural Network', accuracy: 96, status: 'Active' },
    { name: 'SVM', accuracy: 85, status: 'Inactive' },
  ];

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Period Selector */}
      <View style={styles.periodSelector}>
        {periods.map((period) => (
          <TouchableOpacity
            key={period}
            style={[
              styles.periodButton,
              selectedPeriod === period && styles.selectedPeriod,
            ]}
            onPress={() => setSelectedPeriod(period)}
          >
            <Text
              style={[
                styles.periodText,
                selectedPeriod === period && styles.selectedPeriodText,
              ]}
            >
              {period}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Chart */}
      <View style={styles.chartContainer}>
        <Text style={styles.chartTitle}>Performance Analytics</Text>
        <View style={styles.chart}>
          {chartData.map((item, index) => (
            <View key={index} style={styles.chartItem}>
              <View style={styles.barContainer}>
                <View
                  style={[
                    styles.bar,
                    {
                      height: `${item.value}%`,
                      backgroundColor: item.color,
                    },
                  ]}
                />
              </View>
              <Text style={styles.chartLabel}>{item.label}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Metrics */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Key Metrics</Text>
        <View style={styles.metricsGrid}>
          {metrics.map((metric, index) => (
            <View key={index} style={styles.metricCard}>
              <Text style={styles.metricTitle}>{metric.title}</Text>
              <Text style={styles.metricValue}>{metric.value}</Text>
              <View style={styles.metricChange}>
                <Text
                  style={[
                    styles.changeText,
                    { color: metric.positive ? '#10b981' : '#ef4444' },
                  ]}
                >
                  {metric.change}
                </Text>
                <Text style={styles.changeIcon}>
                  {metric.positive ? '↗️' : '↘️'}
                </Text>
              </View>
            </View>
          ))}
        </View>
      </View>

      {/* Models */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>ML Models</Text>
        {models.map((model, index) => (
          <View key={index} style={styles.modelCard}>
            <View style={styles.modelInfo}>
              <Text style={styles.modelName}>{model.name}</Text>
              <View style={styles.modelStats}>
                <Text style={styles.accuracyText}>
                  Accuracy: {model.accuracy}%
                </Text>
                <View
                  style={[
                    styles.statusBadge,
                    {
                      backgroundColor:
                        model.status === 'Active'
                          ? '#10b981'
                          : model.status === 'Training'
                          ? '#f59e0b'
                          : '#6b7280',
                    },
                  ]}
                >
                  <Text style={styles.statusText}>{model.status}</Text>
                </View>
              </View>
            </View>
            <View style={styles.accuracyBar}>
              <View
                style={[
                  styles.accuracyFill,
                  { width: `${model.accuracy}%` },
                ]}
              />
            </View>
          </View>
        ))}
      </View>

      {/* Data Sources */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Data Sources</Text>
        <View style={styles.dataSourcesContainer}>
          <TouchableOpacity style={styles.dataSourceCard}>
            <Text style={styles.dataSourceIcon}>📊</Text>
            <Text style={styles.dataSourceTitle}>CSV Files</Text>
            <Text style={styles.dataSourceCount}>12 files</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.dataSourceCard}>
            <Text style={styles.dataSourceIcon}>🗄️</Text>
            <Text style={styles.dataSourceTitle}>Database</Text>
            <Text style={styles.dataSourceCount}>3 tables</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.dataSourceCard}>
            <Text style={styles.dataSourceIcon}>🌐</Text>
            <Text style={styles.dataSourceTitle}>API</Text>
            <Text style={styles.dataSourceCount}>5 endpoints</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  periodSelector: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 16,
    justifyContent: 'space-between',
  },
  periodButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#e5e7eb',
  },
  selectedPeriod: {
    backgroundColor: '#6366f1',
  },
  periodText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  selectedPeriodText: {
    color: '#ffffff',
  },
  chartContainer: {
    backgroundColor: '#ffffff',
    margin: 20,
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  chartTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 20,
  },
  chart: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    height: 120,
  },
  chartItem: {
    alignItems: 'center',
    flex: 1,
  },
  barContainer: {
    height: 100,
    width: 24,
    justifyContent: 'flex-end',
    marginBottom: 8,
  },
  bar: {
    width: '100%',
    borderRadius: 12,
    minHeight: 4,
  },
  chartLabel: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '500',
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    width: (width - 60) / 2,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  metricTitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4,
  },
  metricChange: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  changeText: {
    fontSize: 14,
    fontWeight: '600',
    marginRight: 4,
  },
  changeIcon: {
    fontSize: 12,
  },
  modelCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  modelInfo: {
    marginBottom: 12,
  },
  modelName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  modelStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  accuracyText: {
    fontSize: 14,
    color: '#6b7280',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#ffffff',
  },
  accuracyBar: {
    height: 6,
    backgroundColor: '#e5e7eb',
    borderRadius: 3,
  },
  accuracyFill: {
    height: '100%',
    backgroundColor: '#6366f1',
    borderRadius: 3,
  },
  dataSourcesContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  dataSourceCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  dataSourceIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  dataSourceTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  dataSourceCount: {
    fontSize: 12,
    color: '#6b7280',
  },
});

export default DataScreen;