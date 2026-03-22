import { motion } from 'framer-motion';

export const SystemManifesto = () => {
  return (
    <motion.div 
      onClick={(e) => e.stopPropagation()} // Shield
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-[#e3bf17]/30 p-20 rounded-2xl flex flex-col items-center justify-center"
    >
      <h2 className="font-jersey text-7xl text-[#e3bf17] mb-6 uppercase">
        PROJECT_CHEQUITA
      </h2>
      
      <div className="space-y-6 font-mono text-lg text-[#e3bf17]/90 text-center max-w-2xl">
        <p className="tracking-widest">[STATUS] ACTIVE_QUANT_LINK</p>
        <p>
          Chequita turns alpha ideas into actionable trades: precise 
          entries/exits, smart sizing, realistic holds explained clearly
          so you trust and execute with confidence.
        </p>
        <div className="pt-4 border-t border-[#e3bf17]/10 w-full">
          <p className="text-sm italic opacity-60">
            "Hunt the Alpha Before It Grows."
          </p>
        </div>
      </div>
    </motion.div>
  );
};