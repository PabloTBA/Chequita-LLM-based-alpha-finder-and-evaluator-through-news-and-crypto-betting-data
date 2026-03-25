import { NavButton } from './NavButton';

interface SidebarProps {
  setActiveTab: (tab: string) => void;
}

export const Sidebar = ({ setActiveTab }: SidebarProps) => {
  return (
    <nav className="fixed left-0 top-0 h-screen flex flex-col justify-center p-10 z-[60] pointer-events-none">
      <div className="pointer-events-auto flex flex-col gap-4">
        <NavButton label="Generate" onClick={() => setActiveTab('generate')} />
        <NavButton label="Analysis" onClick={() => setActiveTab('analysis')} />
        <NavButton label="About Us" onClick={() => setActiveTab('about')} />
      </div>
      <div className="absolute left-8 top-1/4 bottom-1/4 w-[1px] bg-[#e3bf17]/20 -z-10" />
    </nav>
  );
};