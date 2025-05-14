import React, { useState } from 'react';

interface PasswordInputProps {
  onSubmit: (password: string) => void;
  placeholder?: string;
  buttonText?: string;
}

const PasswordInput: React.FC<PasswordInputProps> = ({
  onSubmit,
  placeholder = 'Enter secure access code',
  buttonText = 'VERIFY',
}) => {
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (password.trim()) {
      onSubmit(password);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="matrix-input-container mt-4">
      <span className="matrix-input-prefix">SECURE&gt;</span>
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="matrix-input secure-input"
        placeholder={placeholder}
        autoComplete="off"
      />
      <button
        type="submit"
        className="px-4 py-2 hover:bg-green-900 hover:bg-opacity-30"
        disabled={!password.trim()}
      >
        {buttonText}
      </button>
    </form>
  );
};

export default PasswordInput; 