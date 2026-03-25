import { motion } from 'framer-motion';

const members = [
  { id: "0x01", name: "Enzo Panaligan", role: "Lead Architect / Back-end Designer / 皮仁杰" },
  { id: "0x02", name: "Viggo Kristoffer Araneta", role: "Front-end Designer / Marksman" },
  { id: "0x03", name: "Gilpaul Miguel Dela Cruz", role: "Quant Researcher / Frost Mage" },
  { id: "0x04", name: "Pio Renato Sanchez", role: "Quant Researcher / Shadow Operative" },
  { id: "0x05", name: "Alfred Tancotian", role: "SYSTEM ERROR: FILE TOO LARGE" },
];

export const Creators = () => {
  return (
    <motion.div 
      onClick={(e) => e.stopPropagation()} // Shield
      initial={{ opacity: 0, x: -30 }}
      animate={{ opacity: 1, x: 0 }}
      className="max-w-4xl w-full bg-black/60 backdrop-blur-xl border-l-4 border-[#e3bf17] p-16 rounded-r-2xl"
    >
      <h2 className="font-jersey text-7xl text-[#e3bf17] mb-12 uppercase animate-neon-buzz text-left">
        CREATOR_DIRECTORY
      </h2>
      
      <div className="flex flex-col gap-8">
        {members.map((m, i) => (
          <motion.div 
            key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
            className="group flex items-center gap-8 border-b border-[#e3bf17]/10 pb-4 hover:border-[#e3bf17]/40 transition-colors"
          >
            <span className="font-mono text-[#e3bf17] opacity-40 text-sm">{m.id}</span>
            <div className="flex flex-col items-start">
              <p className="font-jersey text-4xl text-[#e3bf17] m-0 leading-none group-hover:drop-shadow-[0_0_8px_rgba(227,191,23,0.5)] transition-all">
                {m.name}
              </p>
              <p className="font-mono text-[10px] text-zinc-500 uppercase tracking-[0.4em] mt-2">
                {m.role}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};