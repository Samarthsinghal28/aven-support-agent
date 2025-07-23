// frontend/src/components/vapi-voice-call.tsx
"use client";
import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import useVapi, { VapiState } from '@/hooks/use-vapi';

interface VapiVoiceCallProps {
  assistantId: string;
}

const VapiVoiceCall: React.FC<VapiVoiceCallProps> = ({ assistantId }) => {
  const { vapiState, volumeLevel, toggleCall } = useVapi();
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const uniformsRef = useRef<any>(null);
  // Track requestAnimationFrame so we can cancel it on unmount
  const animationIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // --- Guard against duplicate canvases ---------------------------------
    // If React remounts this component (e.g. Strict Mode double-mount or hot
    // reload) there might already be a canvas inside the mount element. Make
    // sure we start with a clean slate to avoid rendering multiple orbs.
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild);
    }
    // ----------------------------------------------------------------------

    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
    camera.position.set(0, 0, 10);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Orb object
    const uniforms = {
      u_time: { value: 0.0 },
      u_frequency: { value: 0.0 },
      u_red: { value: 0.4 },
      u_green: { value: 0.6 },
      u_blue: { value: 1.0 },
    };
    uniformsRef.current = uniforms;

    const mat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: `
        uniform float u_time;
        uniform float u_frequency;
        varying float v_noise;

        // Perlin Noise
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 permute(vec4 x) { return mod289(((x*34.0)+10.0)*x); }
        vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
        float snoise(vec3 v) {
          const vec2 C = vec2(1.0/6.0, 1.0/3.0);
          const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
          vec3 i = floor(v + dot(v, C.yyy));
          vec3 x0 = v - i + dot(i, C.xxx);
          vec3 g = step(x0.yzx, x0.xyz);
          vec3 l = 1.0 - g;
          vec3 i1 = min(g.xyz, l.zxy);
          vec3 i2 = max(g.xyz, l.zxy);
          vec3 x1 = x0 - i1 + C.xxx;
          vec3 x2 = x0 - i2 + C.yyy;
          vec3 x3 = x0 - D.yyy;
          i = mod289(i);
          vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
          float n_ = 0.142857142857;
          vec3 ns = n_ * D.wyz - D.xzx;
          vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
          vec4 x_ = floor(j * ns.z);
          vec4 y_ = floor(j - 7.0 * x_);
          vec4 x = x_ * ns.x + ns.yyyy;
          vec4 y = y_ * ns.x + ns.yyyy;
          vec4 h = 1.0 - abs(x) - abs(y);
          vec4 b0 = vec4(x.xy, y.xy);
          vec4 b1 = vec4(x.zw, y.zw);
          vec4 s0 = floor(b0) * 2.0 + 1.0;
          vec4 s1 = floor(b1) * 2.0 + 1.0;
          vec4 sh = -step(h, vec4(0.0));
          vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
          vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
          vec3 p0 = vec3(a0.xy, h.x);
          vec3 p1 = vec3(a0.zw, h.y);
          vec3 p2 = vec3(a1.xy, h.z);
          vec3 p3 = vec3(a1.zw, h.w);
          vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
          p0 *= norm.x;
          p1 *= norm.y;
          p2 *= norm.z;
          p3 *= norm.w;
          vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
          m = m * m;
          return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
        }

        void main() {
          float noise = snoise(position + u_time * 0.1);
          float displacement = (u_frequency / 20.0) * noise;
          v_noise = noise;
          vec3 newPosition = position + normal * (0.5 + displacement);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
        }
      `,
      fragmentShader: `
        uniform float u_red;
        uniform float u_green;
        uniform float u_blue;
        varying float v_noise;
        
        void main() {
          float intensity = 0.5 + 0.5 * v_noise;
          gl_FragColor = vec4(u_red * intensity, u_green * intensity, u_blue * intensity, 1.0);
        }
      `,
      wireframe: true,
    });

    const geo = new THREE.IcosahedronGeometry(2, 20);
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    meshRef.current = mesh;

    // Animation loop
    const clock = new THREE.Clock();
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      if (uniformsRef.current && meshRef.current) {
        uniformsRef.current.u_time.value = clock.getElapsedTime();
        uniformsRef.current.u_frequency.value = THREE.MathUtils.lerp(uniformsRef.current.u_frequency.value, volumeLevel, 0.2);
        meshRef.current.rotation.y += 0.001;
      }
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (mountRef.current && rendererRef.current && cameraRef.current) {
        cameraRef.current.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      // Cancel the animation loop
      if (animationIdRef.current !== null) {
        cancelAnimationFrame(animationIdRef.current);
      }

      // Dispose of Three.js objects to free GPU memory
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }

      if (mountRef.current) {
        // Remove any remaining canvases (safety net)
        while (mountRef.current.firstChild) {
          mountRef.current.removeChild(mountRef.current.firstChild);
        }
      }
    };
  }, []);

  const getStatusText = (state: VapiState) => {
    switch (state) {
      case 'idle': return 'Click to start call';
      case 'connecting': return 'Connecting...';
      case 'connected': return 'Connected. Say something!';
      case 'listening': return 'Listening...';
      case 'speaking': return 'Agent is speaking...';
      case 'error': return 'Error. Click to retry.';
      default: return '';
    }
  };

  return (
    <div className="relative w-full h-96 flex flex-col items-center justify-center cursor-pointer" onClick={() => toggleCall(assistantId)}>
      <div ref={mountRef} className="absolute inset-0" />
      <div className="absolute text-white text-center z-10 pointer-events-none">
        <h2 className="text-2xl font-bold">{getStatusText(vapiState)}</h2>
        {vapiState === 'connected' && <p className="text-sm">Speak now to talk with the assistant.</p>}
      </div>
    </div>
  );
};

export default VapiVoiceCall; 