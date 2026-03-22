import { motion } from 'framer-motion';

interface SummaryProps {
  onBack: () => void;
}

export const TraderSummary = ({ onBack }: SummaryProps) => {
  return (
    <motion.div 
      initial={{ x: '100vw' }} // Starts off-screen to the right
      animate={{ x: 0 }}      // Slides in to the center
      exit={{ x: '100vw' }}    // Slides back out to the right when dismissed
      transition={{ type: "spring", damping: 25, stiffness: 120 }}
      className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 backdrop-blur-3xl z-[100] p-20"
    >
      {/* Back Button - Moved slightly down so it doesn't hit the header area */}
      <button 
        onClick={onBack}
        className="absolute top-12 left-12 font-jersey text-4xl text-[#e3bf17] hover:tracking-[0.2em] transition-all uppercase flex items-center gap-4 group"
      >
        <span className="group-hover:-translate-x-2 transition-transform">{'<<<'}</span> 
        RETURN_TO_SYSTEM_HUB
      </button>

      <div className="max-w-6xl w-full border-2 border-[#e3bf17]/40 p-20 rounded-3xl bg-black/50 shadow-[0_0_50px_rgba(227,191,23,0.1)]">
        <h2 className="font-jersey text-9xl text-[#e3bf17] mb-8 uppercase text-center drop-shadow-[0_0_15px_rgba(227,191,23,0.4)]">
          REPORT_v1.0
        </h2>
        <div className="font-mono text-[#e3bf17]/70 text-center space-y-6 text-xl">
          <p className="animate-pulse">[!] NO_ACTIVE_DATA_STREAM</p>
          <p className="opacity-40">Awaiting input from Generate_Module...</p>
        </div>
      </div>
    </motion.div>
  );
};