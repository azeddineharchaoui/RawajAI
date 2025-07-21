import { Image } from 'expo-image';
import { ActivityIndicator, StyleSheet, View, ScrollView, TouchableOpacity, ImageBackground, Dimensions } from 'react-native';
import { useState, useEffect } from 'react';
import { useRouter } from 'expo-router';

import ParallaxScrollView from '@/components/ParallaxScrollView';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Card } from '@/components/ui/Card';
import { api } from '@/services/api';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import ApiConeection from '@/components/home/ApiConeection';
import StatCard from '@/components/home/stateCard';
import ActionButton from '@/components/home/QuickActions';

export default function DashboardScreen() {
  const router = useRouter();
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];
  const [loading, setLoading] = useState(true);
  const [tunnelStatus, setTunnelStatus] = useState<{ status: string, url?: string } | null>(null);
  const [dashboardData, setDashboardData] = useState({
    pendingAlerts: 3,
    totalProducts: 15,
    lowStockItems: 4,
    upcomingDeliveries: 2,
    anomalyCount: 2
  });

  useEffect(() => {
    const fetchTunnelStatus = async () => {
      try {
        const status = await api.getTunnelStatus();
        setTunnelStatus(status);
      } catch (error) {
        console.error('Error fetching tunnel status:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTunnelStatus();
  }, []);

  const startTunnel = async () => {
    setLoading(true);
    try {
      const result = await api.startTunnel();
      setTunnelStatus(result);
    } catch (error) {
      console.error('Error starting tunnel:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#0a7ea4', dark: '#0a4f66' }}
      headerTitle="Supply Chain AI Dashboard"
      headerImage={
        <ImageBackground
          source={require('@/assets/images/cover.jpeg')}
          style={styles.logo}
          resizeMode='cover'
        />
      }>

      {/* API Connection Status */}
      <ApiConeection />

      {/* Dashboard Overview */}

      {/* Overview Cards */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.cardsScroll}>
        <View style={{ flexDirection: 'row', paddingVertical: 16, justifyContent: 'space-between' }}>
          <StatCard
            title="Products"
            value={dashboardData.totalProducts.toString()}
            icon={require('@/assets/images/product.png')}
            onPress={() => router.push('/(tabs)/inventory')}
          />
          <StatCard
            title="Low Stock"
            value={dashboardData.lowStockItems.toString()}
            icon={require('@/assets/images/lowStock.png')}
            color="#E53E3E"
            onPress={() => router.push('/(tabs)/inventory')}
          />
          <StatCard
            title="Anomalies"
            value={dashboardData.anomalyCount.toString()}
            icon={require('@/assets/images/anomalies.png')}
            color="#DD6B20"
            onPress={() => router.push('/(tabs)/analytics')}
          />
          <StatCard
            title="Alerts"
            value={dashboardData.pendingAlerts.toString()}
            icon={require('@/assets/images/Alerts.png')}
            color="#3182CE"
            onPress={() => router.push('/(tabs)')}
          />
        </View>
      </ScrollView>

      {/* Quick Actions */}
      <ThemedView style={styles.sectionContainer}>
        <ThemedText type="subtitle">Quick Actions</ThemedText>
        <View style={styles.actionGrid}>
          <ActionButton
            title="Demand Forecast"
            icon={require('@/assets/images/Forecast.png')}
            onPress={() => router.push('/(tabs)/forecast')}
          />
          <ActionButton
            title="Inventory Optimization"
            icon={require('@/assets/images/optimisation.png')}
            onPress={() => router.push('/(tabs)/inventory')}
          />
          <ActionButton
            title="Ask Assistant"
            icon={require('@/assets/images/assistent.png')}
            onPress={() => router.push('/(tabs)/assistant')}
          />
          <ActionButton
            title="Generate Report"
            icon={require('@/assets/images/reporti.webp')}
            onPress={() => router.push('/(tabs)/analytics')}
          />
        </View>
      </ThemedView>

      {/* Recent Activity */}
      <ThemedView style={styles.sectionContainer}>
        <ThemedText type="subtitle">Recent Activity</ThemedText>
        <ActivityItem
          title="Anomaly Detected"
          description="Unusual demand pattern for product ID: SM-5432"
          time="2h ago"
        />
        <ActivityItem
          title="Forecast Updated"
          description="Quarterly demand forecast has been updated"
          time="5h ago"
        />
        <ActivityItem
          title="Inventory Optimized"
          description="Stock levels have been optimized for 12 products"
          time="1d ago"
        />
      </ThemedView>
    </ParallaxScrollView>
  );
}


// Activity Item Component
const ActivityItem = ({ title, description, time }: {
  title: string;
  description: string;
  time: string;
}) => {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];

  return (
    <ThemedView style={styles.activityItem}>
      <View style={styles.activityDot} />
      <View style={styles.activityContent}>
        <ThemedText type="defaultSemiBold">{title}</ThemedText>
        <ThemedText style={styles.activityDesc}>{description}</ThemedText>
        <ThemedText style={styles.activityTime}>{time}</ThemedText>
      </View>
    </ThemedView>
  );
};


const styles = StyleSheet.create({
  logo: {
    flex: 1,
    width: '100%',
    height: "100%",
  },
  connectionCard: {
    marginBottom: 16,
    padding: 16,
  },
  loader: {
    marginTop: 8,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 4,
  },
  statusIndicator: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  statusActive: {
    backgroundColor: '#38A169',
  },
  statusInactive: {
    backgroundColor: '#E53E3E',
  },
  urlText: {
    fontSize: 12,
    opacity: 0.7,
  },
  connectButton: {
    marginTop: 8,
  },
  cardsScroll: {
    width: '100%',
    marginHorizontal: -16,
    paddingHorizontal: 16,
    marginBottom: 16,
  },
  statCard: {
    width: 120,
    height: 120,
    padding: 12,
    marginRight: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  icon: {
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  statTitle: {
    fontSize: 12,
    opacity: 0.7,
    textAlign: 'center',
  },
  sectionContainer: {
    marginBottom: 24,
    gap: 16,
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -8,
  },
  actionButton: {
    width: '50%',
    padding: 8,
    alignItems: 'center',
  },
  actionIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  actionTitle: {
    textAlign: 'center',
    fontSize: 14,
  },
  activityItem: {
    flexDirection: 'row',
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(150, 150, 150, 0.2)',
  },
  activityDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#3182CE',
    marginTop: 6,
    marginRight: 12,
  },
  activityContent: {
    flex: 1,
  },
  activityDesc: {
    opacity: 0.7,
    marginTop: 2,
    fontSize: 14,
  },
  activityTime: {
    fontSize: 12,
    opacity: 0.5,
    marginTop: 4,
  },
});
