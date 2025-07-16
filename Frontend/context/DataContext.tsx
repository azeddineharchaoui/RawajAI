import { createContext, useContext, useState, ReactNode } from 'react';
import { ForecastData, Anomaly, ScenarioType } from '@/types/api';

interface DataContextType {
  selectedProduct: string | null;
  forecastData: ForecastData[] | null;
  anomalies: Anomaly[] | null;
  scenario: ScenarioType | null;
  selectProduct: (id: string) => void;
  setForecastData: (data: ForecastData[] | null) => void;
  setAnomalies: (data: Anomaly[] | null) => void;
  setScenario: (type: ScenarioType | null) => void;
}

const DataContext = createContext<DataContextType>({
  selectedProduct: null,
  forecastData: null,
  anomalies: null,
  scenario: null,
  selectProduct: () => {},
  setForecastData: () => {},
  setAnomalies: () => {},
  setScenario: () => {},
});

export const DataProvider = ({ children }: { children: ReactNode }) => {
  const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
  const [forecastData, setForecastData] = useState<ForecastData[] | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[] | null>(null);
  const [scenario, setScenario] = useState<ScenarioType | null>(null);

  const selectProduct = (id: string) => {
    setSelectedProduct(id);
  };

  const value = {
    selectedProduct,
    forecastData,
    anomalies,
    scenario,
    selectProduct,
    setForecastData,
    setAnomalies,
    setScenario,
  };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
};

export const useData = () => useContext(DataContext);
