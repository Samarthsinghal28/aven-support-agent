// frontend/src/hooks/use-vapi.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import Vapi from '@vapi-ai/web';

const VAPI_PUBLIC_KEY = process.env.NEXT_PUBLIC_VAPI_PUBLIC_KEY;

export type VapiState = 'idle' | 'connecting' | 'connected' | 'listening' | 'speaking' | 'error';

const useVapi = () => {
  const [vapiState, setVapiState] = useState<VapiState>('idle');
  const [volumeLevel, setVolumeLevel] = useState(0);
  const vapiRef = useRef<any>(null);

  const initializeVapi = useCallback(() => {
    if (!VAPI_PUBLIC_KEY) {
      console.error('Vapi public key not found. Please set NEXT_PUBLIC_VAPI_PUBLIC_KEY in your .env.local file.');
      setVapiState('error');
      return;
    }

    if (!vapiRef.current) {
      const vapiInstance = new Vapi(VAPI_PUBLIC_KEY);
      vapiRef.current = vapiInstance;

      vapiInstance.on('call-start', () => {
        setVapiState('connected');
      });

      vapiInstance.on('call-end', () => {
        setVapiState('idle');
        setVolumeLevel(0);
      });

      vapiInstance.on('speech-start', () => {
        setVapiState('speaking');
      });
      
      vapiInstance.on('speech-end', () => {
        setVapiState('listening');
      });

      vapiInstance.on('volume-level', (volume: number) => {
        setVolumeLevel(volume);
      });

      vapiInstance.on('error', (e: Error) => {
        console.error('Vapi error:', e);
        setVapiState('error');
      });
    }
  }, []);

  useEffect(() => {
    initializeVapi();

    return () => {
      if (vapiRef.current) {
        vapiRef.current.stop();
        vapiRef.current = null;
      }
    };
  }, [initializeVapi]);

  const toggleCall = async (assistantId?: string) => {
    if (!vapiRef.current) return;

    try {
      if (vapiState !== 'idle' && vapiState !== 'error') {
        await vapiRef.current.stop();
      } else if (assistantId) {
        setVapiState('connecting');
        await vapiRef.current.start(assistantId);
      }
    } catch (err) {
      console.error('Error toggling Vapi session:', err);
      setVapiState('error');
    }
  };

  return { vapiState, volumeLevel, toggleCall };
};

export default useVapi; 