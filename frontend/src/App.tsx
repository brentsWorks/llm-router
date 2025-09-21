import { useState } from 'react';
import { Router, About, Models } from './components/features';


function App() {
  const [currentTab, setCurrentTab] = useState<'router' | 'about' | 'models'>('router');
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
                      onClick={() => setCurrentTab('router')}
                      className={`px-5 py-2.5 text-sm font-light rounded-xl transition-all duration-300 ${
                        currentTab === 'router'
                          ? 'bg-slate-600 text-white shadow-lg'
                          : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      Router
                    </button>
                    <button
                      onClick={() => setCurrentTab('models')}
                      className={`px-5 py-2.5 text-sm font-light rounded-xl transition-all duration-300 ${
                        currentTab === 'models'
                          ? 'bg-slate-600 text-white shadow-lg'
                          : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      Models
                    </button>
                    <button
                      onClick={() => setCurrentTab('about')}
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
                <div className="w-12 h-12 bg-slate-50 rounded-lg flex items-center justify-center mx-auto mb-2 border border-slate-200">
                  <div className="w-6 h-6 bg-slate-300 rounded-sm animate-pulse"></div>
                </div>
                <p className="text-xs text-slate-400 font-light animate-pulse">Ready to route</p>
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
          />
        )}
      </div>
    </div>
  );
}

export default App;
