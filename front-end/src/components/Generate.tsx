import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export const Generate = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [dots, setDots] = useState('');

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isGenerating) {
      interval = setInterval(() => {
        // Cycles: "" -> "." -> ".." -> "..." -> ""
        setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
      }, 500);
    } else {
      setDots('');
    }

    return () => clearInterval(interval);
  }, [isGenerating]);

  return (
    <motion.div 
      onClick={(e) => e.stopPropagation()} // Shielded from click-away logic
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-[#e3bf17]/30 p-20 rounded-2xl flex flex-col items-center justify-center"
    >
      {!isGenerating ? (
        <>
          <h2 className="font-jersey text-7xl text-[#e3bf17] mb-8 uppercase animate-neon-buzz text-center">
            System_Ready
          </h2>
          <button 
            onClick={() => setIsGenerating(true)}
            className="group relative px-12 py-4 border-2 border-[#e3bf17] overflow-hidden transition-all hover:shadow-[0_0_20px_rgba(227,191,23,0.6)]"
          >
            <span className="relative z-10 font-jersey text-4xl text-[#e3bf17] group-hover:text-black transition-colors duration-300 uppercase">
              Execute_Generation
            </span>
            <div className="absolute inset-0 bg-[#e3bf17] translate-y-full group-hover:translate-y-0 transition-transform duration-300 -z-0" />
          </button>
        </>
      ) : (
        <div className="flex flex-col items-center">
          <h2 className="font-jersey text-7xl text-[#e3bf17] mb-4 uppercase w-[600px] text-center">
            Generating{dots}
          </h2>
          <p className="font-mono text-xs text-[#e3bf17]/40 tracking-[0.5em] uppercase animate-pulse mt-4">
            Processing_Neural_Weights // Flux_Stable
          </p>
        </div>
      )}
    </motion.div>
  );
};