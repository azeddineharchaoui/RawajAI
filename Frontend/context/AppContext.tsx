import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { getTunnelUrl, saveTunnelUrl, TunnelService, clearTunnelUrl } from '@/services/api';
import { Language } from '@/types/api';

interface AppContextType {
  isLoading: boolean;
  isConnected: boolean;
  tunnelUrl: string | null;
  language: Language;
  connectToTunnel: () => Promise<boolean>;
  disconnectFromTunnel: () => Promise<boolean>;
  setLanguage: (lang: Language) => void;
}

const AppContext = createContext<AppContextType>({
  isLoading: true,
  isConnected: false,
  tunnelUrl: null,
  language: 'en',
  connectToTunnel: async () => false,
  disconnectFromTunnel: async () => false,
  setLanguage: () => {},
});

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [tunnelUrl, setTunnelUrl] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>('en');

  useEffect(() => {
    const checkConnection = async () => {
      try {
        // Try to get stored tunnel URL
        const storedUrl = await getTunnelUrl();
        setTunnelUrl(storedUrl);

        if (storedUrl) {
          // Test the connection
          const status = await TunnelService.test();
          setIsConnected(status !== null);
        }
      } catch (error) {
        console.error('Failed to check connection:', error);
        setIsConnected(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkConnection();
  }, []);

  const connectToTunnel = async () => {
    try {
      setIsLoading(true);
      const response = await TunnelService.start();

      if (response?.url) {
        await saveTunnelUrl(response.url);
        setTunnelUrl(response.url);
        setIsConnected(true);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to connect to tunnel:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const disconnectFromTunnel = async () => {
    try {
      setIsLoading(true);
      await TunnelService.stop();
      await clearTunnelUrl();
      setTunnelUrl(null);
      setIsConnected(false);
      return true;
    } catch (error) {
      console.error('Failed to disconnect from tunnel:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    isLoading,
    isConnected,
    tunnelUrl,
    language,
    connectToTunnel,
    disconnectFromTunnel,
    setLanguage,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useApp = () => useContext(AppContext);
