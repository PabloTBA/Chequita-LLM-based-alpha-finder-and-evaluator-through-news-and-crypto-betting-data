interface NavButtonProps {
  label: string;
  onClick?: () => void;
}

export const NavButton = ({ label, onClick }: NavButtonProps) => {
  return (
    <button 
      onClick={onClick}
      className="group relative flex items-center gap-4 py-2 px-4 w-64 transition-all duration-300 hover:pl-8"
    >
      {/* The Selection Indicator (Pixel Block) */}
      <div className="w-2 h-8 bg-[#e3bf17] opacity-0 group-hover:opacity-100 transition-opacity" />
      
      {/* The Text */}
      <span className="font-jersey text-4xl text-[#e3bf17] uppercase tracking-widest drop-shadow-[0_0_10px_rgba(227,191,23,0.3)]">
        {label}
      </span>

      {/* Background Glow on Hover */}
      <div className="absolute inset-0 bg-[#e3bf17]/5 -z-10 skew-x-12 origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />
    </button>
  );
};