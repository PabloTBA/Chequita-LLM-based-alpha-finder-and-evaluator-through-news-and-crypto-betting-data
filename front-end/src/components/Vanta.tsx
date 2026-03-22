import React, { useState, useEffect, useRef } from 'react'
import * as THREE from 'three'
// @ts-ignore
import GLOBE from 'vanta/dist/vanta.globe.min'

export const VantaBackground = ({ children }: { children: React.ReactNode }) => {
  const [vantaEffect, setVantaEffect] = useState<any>(null)
  const myRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const vantaFunction = (GLOBE as any).default || GLOBE;
    if (!vantaEffect && myRef.current && typeof vantaFunction === 'function') {
      try {
        setVantaEffect(vantaFunction({
          el: myRef.current,
          THREE: THREE,
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.00,
          scaleMobile: 1.00,
          color: 0xe3bf17, 
          backgroundColor: 0x0,
          size: 1.2
        }))
      } catch (err) {
        console.error("Vanta failed:", err)
      }
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy()
    }
  }, [vantaEffect])

  return (
    <div className="relative min-h-screen w-full">
      <div ref={myRef} className="fixed inset-0 w-full h-full -z-10" />
      <div className="relative z-10 w-full">
        {children}
      </div>
    </div>
  )
}