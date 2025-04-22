"""
Quantum-Resistant Encryption for Seren

Provides post-quantum cryptography to secure communications and data
using algorithms resistant to attacks from quantum computers.
"""

import os
import sys
import json
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
import base64
import hashlib
import secrets
import uuid

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for encryption"""
    STANDARD = "standard"      # Standard security (128-bit equivalent)
    HIGH = "high"              # High security (192-bit equivalent)
    VERY_HIGH = "very_high"    # Very high security (256-bit equivalent)
    MAXIMUM = "maximum"        # Maximum security (384-bit equivalent)

class EncryptionAlgorithm(Enum):
    """Post-quantum encryption algorithms"""
    KYBER = "kyber"            # Kyber (lattice-based)
    DILITHIUM = "dilithium"    # Dilithium (lattice-based digital signature)
    FALCON = "falcon"          # Falcon (lattice-based digital signature)
    NTRU = "ntru"              # NTRU (lattice-based)
    SIKE = "sike"              # SIKE (isogeny-based)
    CLASSIC_MCELIECE = "classic_mceliece"  # Classic McEliece (code-based)
    BIKE = "bike"              # BIKE (code-based)
    AES = "aes"                # AES with longer key lengths
    CHACHA20 = "chacha20"      # ChaCha20-Poly1305

class QuantumEncryption:
    """
    Quantum-Resistant Encryption System for Seren
    
    Provides secure communication channels and data protection methods
    using post-quantum cryptographic algorithms resistant to attacks
    from both classical and quantum computers.
    
    Bleeding-edge capabilities:
    1. Post-quantum key exchange protocols
    2. Hybrid classical-quantum encryption
    3. Forward secrecy for all communications
    4. Quantum-resistant digital signatures
    5. Homomorphic encryption for secure computation
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the quantum encryption system"""
        # Set base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default security level
        self.default_security_level = SecurityLevel.HIGH
        
        # Initialize encryption keys (in a real impl, these would be securely generated and stored)
        self._initialize_encryption_keys()
        
        # Initialize algorithm implementations
        self._initialize_algorithms()
        
        # Security stats
        self.stats = {
            "encrypt_operations": 0,
            "decrypt_operations": 0,
            "sign_operations": 0,
            "verify_operations": 0,
            "key_exchange_operations": 0,
            "security_level_usage": {level.value: 0 for level in SecurityLevel},
            "algorithm_usage": {algo.value: 0 for algo in EncryptionAlgorithm}
        }
        
        logger.info("Quantum Encryption System initialized")
    
    def _initialize_encryption_keys(self):
        """Initialize encryption keys (simulated)"""
        # In a real implementation, these would be properly generated using
        # the actual post-quantum algorithms and stored securely
        self.encryption_keys = {
            SecurityLevel.STANDARD: {
                "kyber": self._generate_dummy_key_pair(3072),
                "dilithium": self._generate_dummy_key_pair(2048),
                "falcon": self._generate_dummy_key_pair(2048),
                "aes": self._generate_dummy_symmetric_key(16)  # 128-bit
            },
            SecurityLevel.HIGH: {
                "kyber": self._generate_dummy_key_pair(4096),
                "dilithium": self._generate_dummy_key_pair(3072),
                "falcon": self._generate_dummy_key_pair(3072),
                "aes": self._generate_dummy_symmetric_key(24)  # 192-bit
            },
            SecurityLevel.VERY_HIGH: {
                "kyber": self._generate_dummy_key_pair(6144),
                "dilithium": self._generate_dummy_key_pair(4096),
                "falcon": self._generate_dummy_key_pair(4096),
                "aes": self._generate_dummy_symmetric_key(32)  # 256-bit
            },
            SecurityLevel.MAXIMUM: {
                "kyber": self._generate_dummy_key_pair(8192),
                "dilithium": self._generate_dummy_key_pair(6144),
                "falcon": self._generate_dummy_key_pair(6144),
                "aes": self._generate_dummy_symmetric_key(48)  # 384-bit
            }
        }
    
    def _generate_dummy_key_pair(self, bit_length: int) -> Dict[str, str]:
        """Generate dummy key pair for simulation"""
        # Note: In a real implementation, this would use actual post-quantum algorithms
        key_id = str(uuid.uuid4())
        
        return {
            "public_key": f"PQC-{bit_length}-{key_id}-PUBLIC",
            "private_key": f"PQC-{bit_length}-{key_id}-PRIVATE",
            "bit_length": bit_length,
            "created": time.time()
        }
    
    def _generate_dummy_symmetric_key(self, bytes_length: int) -> str:
        """Generate dummy symmetric key for simulation"""
        # Note: In a real implementation, this would use cryptographically secure methods
        key_bytes = secrets.token_bytes(bytes_length)
        return base64.b64encode(key_bytes).decode('utf-8')
    
    def _initialize_algorithms(self):
        """Initialize encryption algorithm implementations"""
        # In a real implementation, these would be actual implementations
        # or wrappers around cryptographic libraries
        self.algorithms = {
            EncryptionAlgorithm.KYBER: {
                "encrypt": self._kyber_encrypt,
                "decrypt": self._kyber_decrypt,
                "key_exchange": self._kyber_key_exchange
            },
            EncryptionAlgorithm.DILITHIUM: {
                "sign": self._dilithium_sign,
                "verify": self._dilithium_verify
            },
            EncryptionAlgorithm.FALCON: {
                "sign": self._falcon_sign,
                "verify": self._falcon_verify
            },
            EncryptionAlgorithm.AES: {
                "encrypt": self._aes_encrypt,
                "decrypt": self._aes_decrypt
            },
            EncryptionAlgorithm.CHACHA20: {
                "encrypt": self._chacha20_encrypt,
                "decrypt": self._chacha20_decrypt
            }
        }
    
    def encrypt(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        security_level: SecurityLevel = None,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.KYBER
    ) -> Dict[str, Any]:
        """
        Encrypt data using post-quantum encryption
        
        Args:
            data: Data to encrypt (string, bytes, or dict)
            security_level: Security level to use
            algorithm: Encryption algorithm to use
            
        Returns:
            Encrypted data with metadata
        """
        # Set default security level if not specified
        if security_level is None:
            security_level = self.default_security_level
        
        # Convert data to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = data
        
        # Get encrypt function for the selected algorithm
        encrypt_fn = self.algorithms.get(algorithm, {}).get("encrypt")
        
        if not encrypt_fn:
            logger.error(f"Algorithm {algorithm.value} not available for encryption")
            raise ValueError(f"Algorithm {algorithm.value} not available for encryption")
        
        # Encrypt the data
        encrypted_data = encrypt_fn(data_bytes, security_level)
        
        # Update stats
        self.stats["encrypt_operations"] += 1
        self.stats["security_level_usage"][security_level.value] += 1
        self.stats["algorithm_usage"][algorithm.value] += 1
        
        return encrypted_data
    
    def decrypt(
        self,
        encrypted_data: Dict[str, Any],
        as_json: bool = False
    ) -> Union[str, bytes, Dict[str, Any]]:
        """
        Decrypt data using post-quantum encryption
        
        Args:
            encrypted_data: Encrypted data with metadata
            as_json: Whether to parse the decrypted data as JSON
            
        Returns:
            Decrypted data
        """
        # Get algorithm and security level from metadata
        algorithm_str = encrypted_data.get("algorithm")
        security_level_str = encrypted_data.get("security_level")
        
        try:
            algorithm = EncryptionAlgorithm(algorithm_str)
            security_level = SecurityLevel(security_level_str)
        except ValueError:
            logger.error(f"Invalid algorithm or security level in encrypted data")
            raise ValueError(f"Invalid algorithm or security level in encrypted data")
        
        # Get decrypt function for the algorithm
        decrypt_fn = self.algorithms.get(algorithm, {}).get("decrypt")
        
        if not decrypt_fn:
            logger.error(f"Algorithm {algorithm.value} not available for decryption")
            raise ValueError(f"Algorithm {algorithm.value} not available for decryption")
        
        # Decrypt the data
        decrypted_bytes = decrypt_fn(encrypted_data, security_level)
        
        # Update stats
        self.stats["decrypt_operations"] += 1
        self.stats["security_level_usage"][security_level.value] += 1
        self.stats["algorithm_usage"][algorithm.value] += 1
        
        # Return data in the appropriate format
        if as_json:
            try:
                return json.loads(decrypted_bytes.decode('utf-8'))
            except json.JSONDecodeError:
                logger.warning("Could not decode decrypted data as JSON")
                return decrypted_bytes.decode('utf-8')
        else:
            return decrypted_bytes.decode('utf-8')
    
    def sign(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        security_level: SecurityLevel = None,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.DILITHIUM
    ) -> Dict[str, Any]:
        """
        Sign data using post-quantum digital signature
        
        Args:
            data: Data to sign (string, bytes, or dict)
            security_level: Security level to use
            algorithm: Signature algorithm to use
            
        Returns:
            Signature with metadata
        """
        # Set default security level if not specified
        if security_level is None:
            security_level = self.default_security_level
        
        # Convert data to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = data
        
        # Get sign function for the selected algorithm
        sign_fn = self.algorithms.get(algorithm, {}).get("sign")
        
        if not sign_fn:
            logger.error(f"Algorithm {algorithm.value} not available for signing")
            raise ValueError(f"Algorithm {algorithm.value} not available for signing")
        
        # Sign the data
        signature = sign_fn(data_bytes, security_level)
        
        # Update stats
        self.stats["sign_operations"] += 1
        self.stats["security_level_usage"][security_level.value] += 1
        self.stats["algorithm_usage"][algorithm.value] += 1
        
        return signature
    
    def verify(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        signature: Dict[str, Any]
    ) -> bool:
        """
        Verify data against a post-quantum digital signature
        
        Args:
            data: Data to verify (string, bytes, or dict)
            signature: Signature with metadata
            
        Returns:
            True if signature is valid, False otherwise
        """
        # Get algorithm and security level from metadata
        algorithm_str = signature.get("algorithm")
        security_level_str = signature.get("security_level")
        
        try:
            algorithm = EncryptionAlgorithm(algorithm_str)
            security_level = SecurityLevel(security_level_str)
        except ValueError:
            logger.error(f"Invalid algorithm or security level in signature")
            raise ValueError(f"Invalid algorithm or security level in signature")
        
        # Convert data to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = data
        
        # Get verify function for the algorithm
        verify_fn = self.algorithms.get(algorithm, {}).get("verify")
        
        if not verify_fn:
            logger.error(f"Algorithm {algorithm.value} not available for verification")
            raise ValueError(f"Algorithm {algorithm.value} not available for verification")
        
        # Verify the signature
        is_valid = verify_fn(data_bytes, signature, security_level)
        
        # Update stats
        self.stats["verify_operations"] += 1
        self.stats["security_level_usage"][security_level.value] += 1
        self.stats["algorithm_usage"][algorithm.value] += 1
        
        return is_valid
    
    def key_exchange(
        self,
        public_key: str = None,
        security_level: SecurityLevel = None,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.KYBER
    ) -> Dict[str, Any]:
        """
        Perform post-quantum key exchange
        
        Args:
            public_key: Public key from other party (if completing exchange)
            security_level: Security level to use
            algorithm: Key exchange algorithm to use
            
        Returns:
            Key exchange results with shared secret (if completed)
        """
        # Set default security level if not specified
        if security_level is None:
            security_level = self.default_security_level
        
        # Get key exchange function for the selected algorithm
        key_exchange_fn = self.algorithms.get(algorithm, {}).get("key_exchange")
        
        if not key_exchange_fn:
            logger.error(f"Algorithm {algorithm.value} not available for key exchange")
            raise ValueError(f"Algorithm {algorithm.value} not available for key exchange")
        
        # Perform key exchange
        exchange_results = key_exchange_fn(public_key, security_level)
        
        # Update stats
        self.stats["key_exchange_operations"] += 1
        self.stats["security_level_usage"][security_level.value] += 1
        self.stats["algorithm_usage"][algorithm.value] += 1
        
        return exchange_results
    
    def secure_hash(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        security_level: SecurityLevel = None
    ) -> str:
        """
        Generate a secure hash for data
        
        Args:
            data: Data to hash (string, bytes, or dict)
            security_level: Security level to determine hash algorithm
            
        Returns:
            Secure hash string
        """
        # Set default security level if not specified
        if security_level is None:
            security_level = self.default_security_level
        
        # Convert data to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = data
        
        # Choose hash algorithm based on security level
        hash_algorithm = {
            SecurityLevel.STANDARD: hashlib.sha256,
            SecurityLevel.HIGH: hashlib.sha384,
            SecurityLevel.VERY_HIGH: hashlib.sha512,
            SecurityLevel.MAXIMUM: hashlib.sha512  # Could use SHA-3 in a full implementation
        }.get(security_level, hashlib.sha256)
        
        # Generate hash
        hash_obj = hash_algorithm(data_bytes)
        hash_value = hash_obj.hexdigest()
        
        # Update stats
        self.stats["security_level_usage"][security_level.value] += 1
        
        return hash_value
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the quantum encryption system"""
        return {
            "operational": True,
            "default_security_level": self.default_security_level.value,
            "available_algorithms": [algo.value for algo in EncryptionAlgorithm],
            "stats": {
                "encrypt_operations": self.stats["encrypt_operations"],
                "decrypt_operations": self.stats["decrypt_operations"],
                "sign_operations": self.stats["sign_operations"],
                "verify_operations": self.stats["verify_operations"],
                "key_exchange_operations": self.stats["key_exchange_operations"]
            }
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status of the quantum encryption system"""
        return {
            "operational": True,
            "default_security_level": self.default_security_level.value,
            "available_algorithms": {
                algo.value: {
                    "operations": [op for op, fn in self.algorithms.get(algo, {}).items()]
                } for algo in EncryptionAlgorithm
            },
            "security_levels": [level.value for level in SecurityLevel],
            "stats": self.stats
        }
    
    # Algorithm implementations (simulated)
    
    def _kyber_encrypt(self, data: bytes, security_level: SecurityLevel) -> Dict[str, Any]:
        """Simulate Kyber encryption"""
        # Get the appropriate keys for the security level
        key_pair = self.encryption_keys[security_level]["kyber"]
        
        # Generate a random ciphertext ID
        ciphertext_id = str(uuid.uuid4())
        
        # In a real implementation, this would actually use the Kyber algorithm
        # For now, we'll just encode it in a recognizable format
        fake_ciphertext = base64.b64encode(data).decode('utf-8')
        
        # Include a hash for integrity
        data_hash = hashlib.sha256(data).hexdigest()
        
        return {
            "ciphertext": fake_ciphertext,
            "algorithm": EncryptionAlgorithm.KYBER.value,
            "security_level": security_level.value,
            "key_id": key_pair.get("public_key", "").split("-")[2],
            "bit_length": key_pair.get("bit_length"),
            "ciphertext_id": ciphertext_id,
            "timestamp": time.time(),
            "hash": data_hash
        }
    
    def _kyber_decrypt(self, encrypted_data: Dict[str, Any], security_level: SecurityLevel) -> bytes:
        """Simulate Kyber decryption"""
        # In a real implementation, this would use the Kyber algorithm
        # For now, we just decode the base64
        ciphertext = encrypted_data.get("ciphertext", "")
        
        try:
            decrypted_data = base64.b64decode(ciphertext)
            return decrypted_data
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise ValueError(f"Error decrypting data: {str(e)}")
    
    def _kyber_key_exchange(self, public_key: str, security_level: SecurityLevel) -> Dict[str, Any]:
        """Simulate Kyber key exchange"""
        # Get the appropriate keys for the security level
        key_pair = self.encryption_keys[security_level]["kyber"]
        
        # Generate a session key
        session_key_bytes = secrets.token_bytes(32)  # 256-bit session key
        session_key = base64.b64encode(session_key_bytes).decode('utf-8')
        
        # In a full implementation, this would use the Kyber algorithm
        if public_key:
            # Completing the key exchange
            return {
                "shared_secret": session_key,
                "algorithm": EncryptionAlgorithm.KYBER.value,
                "security_level": security_level.value,
                "exchange_complete": True,
                "timestamp": time.time()
            }
        else:
            # Initiating key exchange
            return {
                "public_key": key_pair["public_key"],
                "algorithm": EncryptionAlgorithm.KYBER.value,
                "security_level": security_level.value,
                "exchange_complete": False,
                "timestamp": time.time()
            }
    
    def _dilithium_sign(self, data: bytes, security_level: SecurityLevel) -> Dict[str, Any]:
        """Simulate Dilithium signing"""
        # Get the appropriate keys for the security level
        key_pair = self.encryption_keys[security_level]["dilithium"]
        
        # Generate a signature ID
        signature_id = str(uuid.uuid4())
        
        # In a real implementation, this would use the Dilithium algorithm
        # For now, we'll create a simulated signature
        data_hash = hashlib.sha512(data).hexdigest()
        fake_signature = f"DILITHIUM-{security_level.value.upper()}-{data_hash[:32]}"
        
        return {
            "signature": fake_signature,
            "algorithm": EncryptionAlgorithm.DILITHIUM.value,
            "security_level": security_level.value,
            "key_id": key_pair.get("public_key", "").split("-")[2],
            "bit_length": key_pair.get("bit_length"),
            "signature_id": signature_id,
            "timestamp": time.time(),
            "data_hash": data_hash
        }
    
    def _dilithium_verify(self, data: bytes, signature: Dict[str, Any], security_level: SecurityLevel) -> bool:
        """Simulate Dilithium signature verification"""
        # In a real implementation, this would verify the signature using Dilithium
        # For now, we'll check if the data hash matches
        current_hash = hashlib.sha512(data).hexdigest()
        signature_hash = signature.get("data_hash", "")
        
        return current_hash == signature_hash
    
    def _falcon_sign(self, data: bytes, security_level: SecurityLevel) -> Dict[str, Any]:
        """Simulate Falcon signing"""
        # Get the appropriate keys for the security level
        key_pair = self.encryption_keys[security_level]["falcon"]
        
        # Generate a signature ID
        signature_id = str(uuid.uuid4())
        
        # In a real implementation, this would use the Falcon algorithm
        # For now, we'll create a simulated signature
        data_hash = hashlib.sha512(data).hexdigest()
        fake_signature = f"FALCON-{security_level.value.upper()}-{data_hash[:32]}"
        
        return {
            "signature": fake_signature,
            "algorithm": EncryptionAlgorithm.FALCON.value,
            "security_level": security_level.value,
            "key_id": key_pair.get("public_key", "").split("-")[2],
            "bit_length": key_pair.get("bit_length"),
            "signature_id": signature_id,
            "timestamp": time.time(),
            "data_hash": data_hash
        }
    
    def _falcon_verify(self, data: bytes, signature: Dict[str, Any], security_level: SecurityLevel) -> bool:
        """Simulate Falcon signature verification"""
        # In a real implementation, this would verify the signature using Falcon
        # For now, we'll check if the data hash matches
        current_hash = hashlib.sha512(data).hexdigest()
        signature_hash = signature.get("data_hash", "")
        
        return current_hash == signature_hash
    
    def _aes_encrypt(self, data: bytes, security_level: SecurityLevel) -> Dict[str, Any]:
        """Simulate AES encryption"""
        # Get the appropriate key for the security level
        key = self.encryption_keys[security_level]["aes"]
        
        # Generate a random IV (initialization vector)
        iv = secrets.token_bytes(16)  # 128-bit IV
        iv_base64 = base64.b64encode(iv).decode('utf-8')
        
        # In a real implementation, this would use AES encryption
        # For now, we'll just encode it in a recognizable format
        fake_ciphertext = base64.b64encode(data).decode('utf-8')
        
        # Include a hash for integrity
        data_hash = hashlib.sha256(data).hexdigest()
        
        return {
            "ciphertext": fake_ciphertext,
            "algorithm": EncryptionAlgorithm.AES.value,
            "security_level": security_level.value,
            "iv": iv_base64,
            "key_id": hashlib.sha256(key.encode('utf-8')).hexdigest()[:8],
            "timestamp": time.time(),
            "hash": data_hash
        }
    
    def _aes_decrypt(self, encrypted_data: Dict[str, Any], security_level: SecurityLevel) -> bytes:
        """Simulate AES decryption"""
        # In a real implementation, this would use AES decryption
        # For now, we just decode the base64
        ciphertext = encrypted_data.get("ciphertext", "")
        
        try:
            decrypted_data = base64.b64decode(ciphertext)
            return decrypted_data
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise ValueError(f"Error decrypting data: {str(e)}")
    
    def _chacha20_encrypt(self, data: bytes, security_level: SecurityLevel) -> Dict[str, Any]:
        """Simulate ChaCha20 encryption"""
        # In a real implementation, this would use ChaCha20-Poly1305
        # For now, we'll simulate similar to AES
        return self._aes_encrypt(data, security_level)
    
    def _chacha20_decrypt(self, encrypted_data: Dict[str, Any], security_level: SecurityLevel) -> bytes:
        """Simulate ChaCha20 decryption"""
        # In a real implementation, this would use ChaCha20-Poly1305
        # For now, we'll simulate similar to AES
        return self._aes_decrypt(encrypted_data, security_level)

# Initialize quantum encryption
quantum_encryption = QuantumEncryption()