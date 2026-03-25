import { motion } from 'framer-motion';

export const AboutUs = () => {
  return (
    <motion.div 
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="max-w-2xl bg-black/40 backdrop-blur-md border border-[#e3bf17]/20 p-8 rounded-lg shadow-2xl"
    >
      <h2 className="font-jersey text-5xl text-[#e3bf17] mb-6 uppercase">
        System_Manifesto
      </h2>

      <div className="space-y-4 font-mono text-sm text-[#e3bf17]/80 leading-relaxed">
        <p>
          <span className="text-[#e3bf17] font-bold">[PROJECT]</span> CHEQUITA v1.0.4
        </p>
        <p>
          Developed as a Medium Frequency Quantitative Analysis (MFQA) engine, 
          Chequita bridges the gap between raw market volatility and actionable 
          algorithmic execution.
        </p>
        <p>
          Our core philosophy is built on three pillars: 
          <br/>— Precision Bitmapping
          <br/>— Real-time Heuristic Analysis
          <br/>— Zero-latency Visualization
        </p>
        <div className="pt-4 border-t border-[#e3bf17]/10">
          <p className="text-xs opacity-50 italic">
            // AUTH_KEY: 77-KG-DEFICIT-PROTOCOL_ACTIVE
          </p>
        </div>
      </div>
    </motion.div>
  );
};