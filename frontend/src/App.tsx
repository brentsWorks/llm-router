import { useState } from 'react';
import { Router, About, Models } from './components/features';


function App() {
  const [currentTab, setCurrentTab] = useState<'router' | 'about' | 'models'>('router');
  const [routerStatus, setRouterStatus] = useState<{
    state: 'ready' | 'routing' | 'routed' | 'executing' | 'executed' | 'error';
    message: string;
  }>({
    state: 'ready',
    message: 'Ready to route'
  });
  const [preferences, setPreferences] = useState({
    cost_weight: 0.3,
    latency_weight: 0.3,
    quality_weight: 0.4,
    max_cost_per_1k_tokens: undefined,
    max_latency_ms: undefined,
    max_context_length: undefined,
    min_safety_level: undefined,
    excluded_providers: [],
    excluded_models: [],
  });

  const handlePreferencesChange = (newPreferences: any) => {
    setPreferences(newPreferences);
  };

  const handleResetPreferences = () => {
    setPreferences({
      cost_weight: 0.33,
      latency_weight: 0.33,
      quality_weight: 0.34,
      max_cost_per_1k_tokens: undefined,
      max_latency_ms: undefined,
      max_context_length: undefined,
      min_safety_level: undefined,
      excluded_providers: [],
      excluded_models: [],
    });
  };

  const handleTabChange = (tab: 'router' | 'about' | 'models') => {
    setCurrentTab(tab);
    // Reset router status when switching to/from router tab
    if (tab === 'router') {
      setRouterStatus({
        state: 'ready',
        message: 'Ready to route'
      });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header - Fixed */}
      <div className="bg-gradient-to-r from-slate-50 via-gray-50 to-zinc-50 border-b border-gray-100 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center space-x-4 mb-3">
                <div className="w-12 h-12 bg-gradient-to-br from-slate-400 to-slate-600 rounded-xl flex items-center justify-center shadow-lg">
                  <span className="text-white text-2xl">🧠</span>
                </div>
                <div>
                  <h1 className="text-3xl font-light bg-gradient-to-r from-slate-700 to-slate-500 bg-clip-text text-transparent">
                    LLM Router
                  </h1>
                  <p className="text-sm text-slate-500 font-light">
                    Intelligent model selection for optimal LLM routing
                  </p>
                </div>
              </div>
              
                  {/* Tab Navigation */}
                  <div className="flex space-x-2 mt-4">
                    <button
                      onClick={() => handleTabChange('router')}
                      className={`px-5 py-2.5 text-sm font-light rounded-xl transition-all duration-300 ${
                        currentTab === 'router'
                          ? 'bg-slate-600 text-white shadow-lg'
                          : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      Router
                    </button>
                    <button
                      onClick={() => handleTabChange('models')}
                      className={`px-5 py-2.5 text-sm font-light rounded-xl transition-all duration-300 ${
                        currentTab === 'models'
                          ? 'bg-slate-600 text-white shadow-lg'
                          : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      Models
                    </button>
                    <button
                      onClick={() => handleTabChange('about')}
                      className={`px-5 py-2.5 text-sm font-light rounded-xl transition-all duration-300 ${
                        currentTab === 'about'
                          ? 'bg-slate-600 text-white shadow-lg'
                          : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      About
                    </button>
                  </div>
            </div>
            
            {/* Status - Only show on Router tab */}
            {currentTab === 'router' && (
              <div className="text-center">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-2 border transition-all duration-300 ${
                  routerStatus.state === 'ready' ? 'bg-slate-50 border-slate-200' :
                  routerStatus.state === 'routing' ? 'bg-blue-50 border-blue-200' :
                  routerStatus.state === 'routed' ? 'bg-green-50 border-green-200' :
                  routerStatus.state === 'executing' ? 'bg-purple-50 border-purple-200' :
                  routerStatus.state === 'executed' ? 'bg-emerald-50 border-emerald-200' :
                  'bg-red-50 border-red-200'
                }`}>
                  <div className={`w-6 h-6 rounded-sm transition-all duration-300 ${
                    routerStatus.state === 'ready' ? 'bg-slate-300' :
                    routerStatus.state === 'routing' ? 'bg-blue-400 animate-pulse' :
                    routerStatus.state === 'routed' ? 'bg-green-400' :
                    routerStatus.state === 'executing' ? 'bg-purple-400 animate-spin' :
                    routerStatus.state === 'executed' ? 'bg-emerald-400' :
                    'bg-red-400'
                  } ${routerStatus.state === 'routing' || routerStatus.state === 'executing' ? 'animate-pulse' : ''}`}>
                    {routerStatus.state === 'executing' && (
                      <div className="w-full h-full border-2 border-white border-t-transparent rounded-sm animate-spin"></div>
                    )}
                  </div>
                </div>
                <p className={`text-xs font-light transition-colors duration-300 ${
                  routerStatus.state === 'ready' ? 'text-slate-400' :
                  routerStatus.state === 'routing' ? 'text-blue-600' :
                  routerStatus.state === 'routed' ? 'text-green-600' :
                  routerStatus.state === 'executing' ? 'text-purple-600' :
                  routerStatus.state === 'executed' ? 'text-emerald-600' :
                  'text-red-600'
                }`}>
                  {routerStatus.message}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        {currentTab === 'about' ? (
          <About />
        ) : currentTab === 'models' ? (
          <Models />
        ) : (
          <Router
            preferences={preferences}
            onPreferencesChange={handlePreferencesChange}
            onResetPreferences={handleResetPreferences}
            onStatusChange={setRouterStatus}
          />
        )}
      </div>
    </div>
  );
}

export default App;
