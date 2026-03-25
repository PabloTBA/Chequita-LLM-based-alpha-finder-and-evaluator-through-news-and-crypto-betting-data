import { useState } from 'react'
import { VantaBackground } from './components/Vanta'
import { Sidebar } from './components/Sidebar'
import { Creators } from './components/Creators'
import { SystemManifesto } from './components/SystemManifesto'
import { Generate } from './components/Generate'
import { Analysis } from './components/Analysis'
import { ReportSummary } from './components/ReportSummary'
import { TraderSummary } from './components/TraderSummary'
import { AnimatePresence, motion } from 'framer-motion';

function App() {
  const [activeTab, setActiveTab] = useState<string | null>(null);

  const isSummaryMode = activeTab === 'report' || activeTab === 'trader';

  const handleHeaderClick = (e: React.MouseEvent) => {
    e.stopPropagation(); 
    setActiveTab(activeTab === 'manifesto' ? null : 'manifesto');
  };

  const dismissActiveTab = () => {
    // Only dismiss if we aren't in a full-screen summary
    if (!isSummaryMode) {
      setActiveTab(null);
    }
  };

  return (
    <VantaBackground>
      {/* Header */}
      <AnimatePresence>
        {!isSummaryMode && (
          <motion.header 
            key="header"
            initial={{ x: '-100vw', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '-100vw', opacity: 0 }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            onClick={handleHeaderClick}
            className="absolute top-0 left-0 p-10 select-none z-50 cursor-pointer group pointer-events-auto"
          >
            <div className="flex items-end gap-6 transition-transform group-hover:scale-105">
              <h1 className="font-jersey text-8xl text-[#e3bf17] leading-none m-0 uppercase animate-neon-buzz">
                CHEQUITA
              </h1>
              <img src="/usagi-chiikawa.gif" alt="Usagi" className="h-28 w-auto" />
            </div>
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.5 }}
              transition={{ delay: 0.8 }}
              className="font-mono text-[#e3bf17] text-[10px] tracking-[0.5em] mt-4 ml-1 uppercase group-hover:opacity-100 transition-opacity"
            >
              CATCH_THE_EDGE_BEFORE_THE_MARKET_DOES
            </motion.p>
          </motion.header>
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <AnimatePresence>
        {!isSummaryMode && (
          <motion.div
            key="sidebar-container"
            initial={{ x: '-100vw', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '-100vw', opacity: 0 }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            className="z-[60]"
          >
            <Sidebar setActiveTab={setActiveTab} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area - THE DISMISSAL ZONE */}
      <AnimatePresence mode="wait">
        {activeTab && (
          <motion.main 
            key="main-content"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={dismissActiveTab} // This now works!
            className="fixed inset-0 z-40 flex items-center justify-center bg-black/20 cursor-default pointer-events-auto"
          >
            <AnimatePresence mode="wait">
              {activeTab === 'manifesto' && <SystemManifesto key="manifesto" />}
              {activeTab === 'about' && <Creators key="about" />}
              {activeTab === 'generate' && <Generate key="generate" />}
              {activeTab === 'analysis' && <Analysis key="analysis" onSelect={setActiveTab} />}
              {activeTab === 'report' && <ReportSummary key="report" onBack={() => setActiveTab('analysis')} />}
              {activeTab === 'trader' && <TraderSummary key="trader" onBack={() => setActiveTab('analysis')} />}
            </AnimatePresence>
          </motion.main>
        )}
      </AnimatePresence>
    </VantaBackground>
  )
}

export default App