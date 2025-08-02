import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
} from 'react-native';

const SettingsScreen = () => {
  const [notifications, setNotifications] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [autoSave, setAutoSave] = useState(true);
  const [analytics, setAnalytics] = useState(true);

  const settingSections = [
    {
      title: 'General',
      items: [
        {
          icon: '🔔',
          title: 'Notifications',
          subtitle: 'Receive push notifications',
          type: 'switch',
          value: notifications,
          onToggle: setNotifications,
        },
        {
          icon: '🌙',
          title: 'Dark Mode',
          subtitle: 'Switch to dark theme',
          type: 'switch',
          value: darkMode,
          onToggle: setDarkMode,
        },
        {
          icon: '💾',
          title: 'Auto Save',
          subtitle: 'Automatically save work',
          type: 'switch',
          value: autoSave,
          onToggle: setAutoSave,
        },
      ],
    },
    {
      title: 'Privacy & Security',
      items: [
        {
          icon: '📊',
          title: 'Analytics',
          subtitle: 'Help improve the app',
          type: 'switch',
          value: analytics,
          onToggle: setAnalytics,
        },
        {
          icon: '🔒',
          title: 'Privacy Policy',
          subtitle: 'View our privacy policy',
          type: 'link',
          onPress: () => Alert.alert('Privacy Policy', 'Opening privacy policy...'),
        },
        {
          icon: '🛡️',
          title: 'Security Settings',
          subtitle: 'Manage your security preferences',
          type: 'link',
          onPress: () => Alert.alert('Security', 'Opening security settings...'),
        },
      ],
    },
    {
      title: 'Data & Storage',
      items: [
        {
          icon: '☁️',
          title: 'Cloud Sync',
          subtitle: 'Sync data across devices',
          type: 'link',
          onPress: () => Alert.alert('Cloud Sync', 'Managing cloud sync settings...'),
        },
        {
          icon: '📁',
          title: 'Export Data',
          subtitle: 'Download your data',
          type: 'link',
          onPress: () => Alert.alert('Export', 'Preparing data export...'),
        },
        {
          icon: '🗑️',
          title: 'Clear Cache',
          subtitle: 'Free up storage space',
          type: 'link',
          onPress: () => Alert.alert('Clear Cache', 'Cache cleared successfully!'),
        },
      ],
    },
    {
      title: 'Support',
      items: [
        {
          icon: '❓',
          title: 'Help & FAQ',
          subtitle: 'Get help and find answers',
          type: 'link',
          onPress: () => Alert.alert('Help', 'Opening help center...'),
        },
        {
          icon: '💬',
          title: 'Contact Support',
          subtitle: 'Get in touch with our team',
          type: 'link',
          onPress: () => Alert.alert('Support', 'Opening contact form...'),
        },
        {
          icon: '⭐',
          title: 'Rate the App',
          subtitle: 'Leave a review in the app store',
          type: 'link',
          onPress: () => Alert.alert('Rate App', 'Opening app store...'),
        },
      ],
    },
  ];

  const appInfo = {
    version: '1.0.0',
    buildNumber: '2024.01',
    lastUpdate: 'January 2024',
  };

  const renderSettingItem = (item, index) => {
    return (
      <TouchableOpacity
        key={index}
        style={styles.settingItem}
        onPress={item.onPress}
        disabled={item.type === 'switch'}
      >
        <View style={styles.settingLeft}>
          <Text style={styles.settingIcon}>{item.icon}</Text>
          <View style={styles.settingContent}>
            <Text style={styles.settingTitle}>{item.title}</Text>
            <Text style={styles.settingSubtitle}>{item.subtitle}</Text>
          </View>
        </View>
        <View style={styles.settingRight}>
          {item.type === 'switch' ? (
            <Switch
              value={item.value}
              onValueChange={item.onToggle}
              trackColor={{ false: '#e5e7eb', true: '#6366f1' }}
              thumbColor={item.value ? '#ffffff' : '#ffffff'}
            />
          ) : (
            <Text style={styles.chevron}>›</Text>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Profile Summary */}
      <View style={styles.profileSummary}>
        <View style={styles.profileAvatar}>
          <Text style={styles.profileAvatarText}>DS</Text>
        </View>
        <View style={styles.profileInfo}>
          <Text style={styles.profileName}>Data Scientist</Text>
          <Text style={styles.profileEmail}>scientist@example.com</Text>
        </View>
        <TouchableOpacity style={styles.editButton}>
          <Text style={styles.editButtonText}>Edit</Text>
        </TouchableOpacity>
      </View>

      {/* Settings Sections */}
      {settingSections.map((section, sectionIndex) => (
        <View key={sectionIndex} style={styles.section}>
          <Text style={styles.sectionTitle}>{section.title}</Text>
          <View style={styles.settingsCard}>
            {section.items.map((item, itemIndex) => (
              <View key={itemIndex}>
                {renderSettingItem(item, itemIndex)}
                {itemIndex < section.items.length - 1 && (
                  <View style={styles.separator} />
                )}
              </View>
            ))}
          </View>
        </View>
      ))}

      {/* App Information */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>App Information</Text>
        <View style={styles.settingsCard}>
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Version</Text>
            <Text style={styles.infoValue}>{appInfo.version}</Text>
          </View>
          <View style={styles.separator} />
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Build Number</Text>
            <Text style={styles.infoValue}>{appInfo.buildNumber}</Text>
          </View>
          <View style={styles.separator} />
          <View style={styles.infoItem}>
            <Text style={styles.infoLabel}>Last Update</Text>
            <Text style={styles.infoValue}>{appInfo.lastUpdate}</Text>
          </View>
        </View>
      </View>

      {/* Sign Out Button */}
      <View style={styles.section}>
        <TouchableOpacity
          style={styles.signOutButton}
          onPress={() => Alert.alert('Sign Out', 'Are you sure you want to sign out?')}
        >
          <Text style={styles.signOutText}>Sign Out</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.bottomSpacing} />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  profileSummary: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    margin: 20,
    padding: 20,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  profileAvatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#6366f1',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  profileAvatarText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4,
  },
  profileEmail: {
    fontSize: 14,
    color: '#6b7280',
  },
  editButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#6366f1',
  },
  editButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6366f1',
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 12,
  },
  settingsCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    justifyContent: 'space-between',
  },
  settingLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingIcon: {
    fontSize: 24,
    marginRight: 16,
    width: 32,
    textAlign: 'center',
  },
  settingContent: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 2,
  },
  settingSubtitle: {
    fontSize: 14,
    color: '#6b7280',
  },
  settingRight: {
    marginLeft: 16,
  },
  chevron: {
    fontSize: 20,
    color: '#9ca3af',
    fontWeight: '300',
  },
  separator: {
    height: 1,
    backgroundColor: '#f3f4f6',
    marginLeft: 72,
  },
  infoItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  infoLabel: {
    fontSize: 16,
    color: '#1f2937',
  },
  infoValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6b7280',
  },
  signOutButton: {
    backgroundColor: '#ef4444',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  signOutText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  bottomSpacing: {
    height: 40,
  },
});

export default SettingsScreen;