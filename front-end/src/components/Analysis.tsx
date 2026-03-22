import { motion } from 'framer-motion';

interface AnalysisProps {
  onSelect: (tab: string) => void;
}

export const Analysis = ({ onSelect }: AnalysisProps) => {
  return (
    <motion.div 
      onClick={(e) => e.stopPropagation()} // Shield
      initial={{ opacity: 0, y: 50 }} 
      animate={{ opacity: 1, y: 0 }} 
      transition={{ duration: 0.4, ease: "easeOut" }} 
      className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-[#e3bf17]/30 p-16 rounded-2xl flex flex-col items-center"
    >
      <h2 className="font-jersey text-6xl text-[#e3bf17] mb-12 animate-neon-buzz uppercase text-center">
        Analysis_Center
      </h2>
      
      <div className="grid grid-cols-2 gap-8 w-full">
        <button 
          onClick={() => onSelect('report')}
          className="group border border-[#e3bf17]/20 p-12 hover:bg-[#e3bf17]/10 transition-all rounded-xl flex flex-col items-center"
        >
          <span className="font-jersey text-4xl text-[#e3bf17] mb-2 uppercase group-hover:scale-110 transition-transform">
            Report Summary
          </span>
          <span className="font-mono text-[10px] text-[#e3bf17]/40 tracking-[0.3em]">
            TOTAL_SUMMARISED_DATA_V1
          </span>
        </button>

        <button 
          onClick={() => onSelect('trader')}
          className="group border border-[#e3bf17]/20 p-12 hover:bg-[#e3bf17]/10 transition-all rounded-xl flex flex-col items-center"
        >
          <span className="font-jersey text-4xl text-[#e3bf17] mb-2 uppercase group-hover:scale-110 transition-transform">
            Trader Summary
          </span>
          <span className="font-mono text-[10px] text-[#e3bf17]/40 tracking-[0.3em]">
            TRADER_METRICS_DATA_V1
          </span>
        </button>
      </div>
    </motion.div>
  );
};