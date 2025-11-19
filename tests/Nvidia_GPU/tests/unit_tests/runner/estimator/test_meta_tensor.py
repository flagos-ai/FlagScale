import os
import sys

from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.estimator.meta_registry import register_model, remove_registry

# Import the module under test
from flagscale.runner.estimator.meta_tensor import MetaTensor, ShardedDim


class TestShardedDim:
    """Test suite for ShardedDim class."""

    def test_init(self):
        """Test initialization with different values."""
        # Basic initialization
        sdim = ShardedDim(10)
        assert sdim.dim == 10
        assert sdim.shard == 1

        # With sharding
        sdim = ShardedDim(10, 2)
        assert sdim.dim == 10
        assert sdim.shard == 2

        # Negative sharding should default to 1
        sdim = ShardedDim(10, -1)
        assert sdim.dim == 10
        assert sdim.shard == 1

    def test_copy(self):
        """Test copy method."""
        sdim = ShardedDim(10, 2)
        copy = sdim.copy()
        assert copy.dim == 10
        assert copy.shard == 2
        assert copy is not sdim

    def test_string_representation(self):
        """Test string representation."""
        # Without sharding
        sdim = ShardedDim(10)
        assert str(sdim) == "ShardedDim(10, 1)"
        assert repr(sdim) == "ShardedDim(10, 1)"

        # With sharding
        sdim = ShardedDim(10, 2)
        assert str(sdim) == "ShardedDim(10, 2)"
        assert repr(sdim) == "ShardedDim(10, 2)"

    def test_add(self):
        """Test addition operation."""
        sdim1 = ShardedDim(10, 2)
        sdim2 = ShardedDim(20, 2)
        result = sdim1 + sdim2
        assert result.dim == 30
        assert result.shard == 2

        # Different shards should raise error
        sdim3 = ShardedDim(20, 5)
        with pytest.raises(ValueError):
            sdim1 + sdim3

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 + 10

    def test_radd(self):
        """Test reverse addition operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            10 + sdim

    def test_sub(self):
        """Test subtraction operation."""
        sdim1 = ShardedDim(20, 2)
        sdim2 = ShardedDim(10, 2)
        result = sdim1 - sdim2
        assert result.dim == 10
        assert result.shard == 2

        # Different shards should raise error
        sdim3 = ShardedDim(10, 5)
        with pytest.raises(ValueError):
            sdim1 - sdim3

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 - 10

    def test_mul(self):
        """Test multiplication operation."""
        sdim1 = ShardedDim(10, 2)
        sdim2 = ShardedDim(20, 2)
        result = sdim1 * sdim2
        assert result.dim == 200  # 10 * 20
        assert result.shard == 2  # Keeps first shard

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 * 10

    def test_rmul(self):
        """Test reverse multiplication operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            10 * sdim

    def test_truediv(self):
        """Test division operation."""
        sdim1 = ShardedDim(12, 2)
        sdim2 = ShardedDim(2, 2)
        result = sdim1 / sdim2
        assert result.dim == 6  # 12 / 2
        assert result.shard == 2  # Sharding preserved

        # Different shards should raise error
        sdim3 = ShardedDim(5, 1)
        with pytest.raises(ValueError):
            sdim1 / sdim3

        # Division by zero should raise error
        sdim_zero = ShardedDim(0, 2)
        with pytest.raises(ZeroDivisionError):
            sdim1 / sdim_zero

        # Non-integer division should raise error
        sdim4 = ShardedDim(3, 1)
        with pytest.raises(ValueError):
            sdim1 / sdim4  # 10 / 3 is not an integer

        # Non-ShardedDim should raise error
        with pytest.raises(TypeError):
            sdim1 / 5

    def test_rtruediv(self):
        """Test reverse division operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            5 / sdim

    def test_floordiv(self):
        """Test floor division operation."""
        sdim1 = ShardedDim(12, 2)
        sdim2 = ShardedDim(2, 2)
        result = sdim1 // sdim2
        assert result.dim == 6  # 12 // 2
        assert result.shard == 2  # Sharding preserved

        # Floor division should be equivalent to true division for integer results
        assert (sdim1 // sdim2).dim == (sdim1 / sdim2).dim
        assert (sdim1 // sdim2).shard == (sdim1 / sdim2).shard

        # Different shards should raise error
        sdim3 = ShardedDim(5, 1)
        with pytest.raises(ValueError):
            sdim1 // sdim3

    def test_rfloordiv(self):
        """Test reverse floor division operation."""
        sdim = ShardedDim(10, 2)
        with pytest.raises(TypeError):
            5 // sdim

    def test_eq(self):
        """Test equality check."""
        sdim1 = ShardedDim(10, 2)
        sdim2 = ShardedDim(10, 2)
        sdim3 = ShardedDim(20, 4)
        sdim4 = ShardedDim(20, 2)

        assert sdim1 == sdim2
        assert sdim1 != sdim3
        assert sdim1 != sdim4
        assert sdim1 != 10


class TestMetaTensor:
    """Test suite for MetaTensor class."""

    def setup_method(self):
        """Set up test environment for each test."""
        # Reset registries before each test
        import flagscale.runner.estimator.meta_registry as meta_registry

        meta_registry._model_registries = {}

        # Register models used in tests
        register_model("default")
        register_model("test_model")

    def test_init(self):
        """Test initialization with different shapes and shard specs."""
        # Basic initialization
        tensor = MetaTensor([2, 4, 4])
        assert tensor.shape == [2, 4, 4]
        assert tensor.shard_spec == [1, 1, 1]

        # With sharding - ensure dimensions are divisible by their shards
        tensor = MetaTensor([2, 4, 4], [1, 2, 4])
        assert tensor.shape == [2, 4, 4]
        assert tensor.shard_spec == [1, 2, 4]

        # Mismatched lengths should raise error
        with pytest.raises(ValueError):
            MetaTensor([2, 4, 4], [1, 2])

        # Dimensions not divisible by shards should raise error
        with pytest.raises(ValueError):
            MetaTensor([2, 3, 4], [1, 2, 1])

    def test_shape_property(self):
        """Test shape property getter and setter."""
        tensor = MetaTensor([2, 4, 4], [1, 2, 4])

        # Check getter
        assert tensor.shape == [2, 4, 4]

        # Test setter - extending shape
        tensor.shape = [2, 4, 4, 6]
        assert tensor.shape == [2, 4, 4, 6]
        assert tensor.shard_spec == [
            1,
            2,
            4,
            1,
        ]  # Original sharding preserved + new dimension with shard=1

        # Test setter - shrinking shape
        tensor.shape = [2, 4]
        assert tensor.shape == [2, 4]
        assert tensor.shard_spec == [1, 2]  # Original sharding preserved for remaining dimensions

        # Test setter with dimension not divisible by shard
        with pytest.raises(ValueError):
            tensor.shape = [2, 3]  # 3 not divisible by shard 2

    def test_shard_spec_property(self):
        """Test shard_spec property getter and setter."""
        tensor = MetaTensor([2, 4, 4])

        # Check getter
        assert tensor.shard_spec == [1, 1, 1]

        # Test setter with divisible dimensions
        tensor.shard_spec = [1, 2, 4]
        assert tensor.shard_spec == [1, 2, 4]

        # Test setter with non-divisible dimensions
        with pytest.raises(ValueError):
            tensor.shard_spec = [1, 3, 1]  # 4 not divisible by 3

        # Mismatched length should raise error
        with pytest.raises(ValueError):
            tensor.shard_spec = [1, 2]

    def test_total_elements(self):
        """Test total_elements method with and without sharding."""
        # Use dimensions divisible by their shards
        tensor = MetaTensor([2, 4, 4], [1, 2, 1])  # Now 4 is divisible by 2

        # Without sharding
        assert tensor.total_elements(apply_sharding=False) == 32  # 2*4*4

        # With sharding
        assert tensor.total_elements(apply_sharding=True) == 16  # 2*(4/2)*4 = 2*2*4 = 16

    def test_unshard(self):
        """Test unshard method."""
        tensor = MetaTensor([10, 20, 30, 40], [2, 4, 2, 1])

        # Test unshard all dimensions
        tensor_copy = tensor.clone()
        tensor_copy.unshard()
        assert tensor_copy.shard_spec == [1, 1, 1, 1]

        # Test unshard single dimension
        tensor_copy = tensor.clone()
        tensor_copy.unshard(1)
        assert tensor_copy.shard_spec == [2, 1, 2, 1]

        # Test unshard with negative index
        tensor_copy = tensor.clone()
        tensor_copy.unshard(-2)
        assert tensor_copy.shard_spec == [2, 4, 1, 1]

        # Test unshard range
        tensor_copy = tensor.clone()
        tensor_copy.unshard(start=1, end=2)
        assert tensor_copy.shard_spec == [2, 1, 1, 1]

        # Test unshard with negative indices in range
        tensor_copy = tensor.clone()
        tensor_copy.unshard(start=0, end=-2)
        assert tensor_copy.shard_spec == [1, 1, 1, 1]

        # Test out-of-bounds indices
        with pytest.raises(IndexError):
            tensor.clone().unshard(4)
        with pytest.raises(IndexError):
            tensor.clone().unshard(start=5)
        with pytest.raises(ValueError):
            tensor.clone().unshard(start=2, end=1)

    def test_len(self):
        """Test __len__ method."""
        tensor = MetaTensor([2, 3, 4])
        assert len(tensor) == 3

    def test_getitem(self):
        """Test __getitem__ method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])

        # Get single dimension
        assert tensor[1].dim == 20
        assert tensor[1].shard == 4

        # Get slice
        slice_tensor = tensor[1:]
        assert slice_tensor.shape == [20, 30]
        assert slice_tensor.shard_spec == [4, 1]

    def test_setitem(self):
        """Test __setitem__ method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])

        # Set single dimension
        tensor[1] = ShardedDim(40, 2)
        assert tensor.shape == [10, 40, 30]
        assert tensor.shard_spec == [2, 2, 1]

        # Set with invalid type
        with pytest.raises(TypeError):
            tensor[1] = 40

        # Set slice
        other_tensor = MetaTensor([50, 60], [1, 2])
        tensor[1:] = other_tensor
        assert tensor.shape == [10, 50, 60]
        assert tensor.shard_spec == [2, 1, 2]

        # Set slice with invalid type
        with pytest.raises(TypeError):
            tensor[1:] = [50, 60]

        # Set slice with mismatched length
        with pytest.raises(ValueError):
            tensor[1:] = MetaTensor([50], [1])

    def test_iter(self):
        """Test __iter__ method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        dims = list(tensor)
        assert len(dims) == 3
        assert dims[0].dim == 10 and dims[0].shard == 2
        assert dims[1].dim == 20 and dims[1].shard == 4
        assert dims[2].dim == 30 and dims[2].shard == 1

    def test_clone(self):
        """Test copy method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        copy_tensor = tensor.clone()

        # Verify it's a different object with the same values
        assert tensor is not copy_tensor
        assert tensor.shape == copy_tensor.shape
        assert tensor.shard_spec == copy_tensor.shard_spec

        # Modify copy should not affect original
        copy_tensor[0] = ShardedDim(40, 1)
        assert tensor.shape == [10, 20, 30]
        assert copy_tensor.shape == [40, 20, 30]

    def test_index(self):
        """Test index method."""
        tensor = MetaTensor([10, 20, 30, 20], [1, 2, 1, 2])

        # Find existing dimension
        idx = tensor.index(ShardedDim(20, 2))
        assert idx == 1

        # Find non-existent dimension
        with pytest.raises(ValueError):
            tensor.index(ShardedDim(20, 3))

        # Find with invalid type
        with pytest.raises(TypeError):
            tensor.index(20)

    def test_eq(self):
        """Test __eq__ method."""
        tensor1 = MetaTensor([10, 20], [2, 1])
        tensor2 = MetaTensor([10, 20], [2, 1])
        tensor3 = MetaTensor([10, 20], [1, 1])
        tensor4 = MetaTensor([10, 30], [2, 1])

        # Equal tensors
        assert tensor1 == tensor2

        # Different sharding
        assert tensor1 != tensor3

        # Different dimensions
        assert tensor1 != tensor4

        # Different type
        assert tensor1 != [10, 20]

    def test_add(self):
        """Test element-wise addition of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])

        # Element-wise addition
        result = tensor1 + tensor2

        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]

        # Incompatible sharding should raise error
        tensor3 = MetaTensor([10, 20, 30], [1, 4, 1])
        with pytest.raises(ValueError):
            tensor1 + tensor3

    def test_sub(self):
        """Test element-wise subtraction of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])

        # Element-wise subtraction
        result = tensor1 - tensor2

        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]

    def test_mul(self):
        """Test element-wise multiplication of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])

        # Element-wise multiplication
        result = tensor1 * tensor2

        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]

        # Incompatible sharding should raise error
        tensor3 = MetaTensor([10, 20, 30], [1, 4, 1])
        with pytest.raises(ValueError):
            tensor1 * tensor3

        # Scalar multiplication should return copy of tensor
        result = tensor1 * 10
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == tensor1.shard_spec
        assert result is not tensor1  # Should be a copy

        # Non-number/tensor should raise TypeError
        with pytest.raises(TypeError):
            tensor1 * "string"

    def test_div(self):
        """Test element-wise division of two tensors with compatible shapes."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])

        # Element-wise division
        result = tensor1 / tensor2

        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]

        # Incompatible sharding should raise error
        tensor3 = MetaTensor([10, 20, 30], [1, 4, 1])
        with pytest.raises(ValueError):
            tensor1 / tensor3

        # Different shapes should raise error
        tensor4 = MetaTensor([10, 20, 40], [2, 4, 1])
        with pytest.raises(ValueError):
            tensor1 / tensor4

        # Scalar division should return copy of tensor
        result = tensor1 / 10
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == tensor1.shard_spec
        assert result is not tensor1  # Should be a copy

        # Division by zero should raise error
        with pytest.raises(ZeroDivisionError):
            tensor1 / 0

        # Non-number/tensor should raise TypeError
        with pytest.raises(TypeError):
            tensor1 / "string"

    def test_rtruediv(self):
        """Test right-side division."""
        tensor1 = MetaTensor([10, 20, 30], [2, 4, 1])
        tensor2 = MetaTensor([10, 20, 30], [2, 4, 1])

        # Element-wise right division (tensor2 / tensor1)
        result = tensor1.__rtruediv__(tensor2)

        # Result should have same shape and sharding
        assert result.shape == [10, 20, 30]
        assert result.shard_spec == [2, 4, 1]

        # Scalar division
        result = tensor1.__rtruediv__(10)
        assert result.shape == tensor1.shape
        assert result.shard_spec == tensor1.shard_spec

        # Non-tensor/number should raise TypeError
        with pytest.raises(TypeError):
            tensor1.__rtruediv__("string")

    def test_unsqueeze(self):
        """Test unsqueeze method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])

        # Test inserting dimension at the beginning
        unsqueezed = tensor.unsqueeze(0)
        assert unsqueezed.shape == [1, 10, 20, 30]
        assert unsqueezed.shard_spec == [1, 2, 4, 1]

        # Test inserting dimension in the middle
        unsqueezed = tensor.unsqueeze(1)
        assert unsqueezed.shape == [10, 1, 20, 30]
        assert unsqueezed.shard_spec == [2, 1, 4, 1]

        # Test inserting dimension at the end
        unsqueezed = tensor.unsqueeze(-1)
        assert unsqueezed.shape == [10, 20, 30, 1]
        assert unsqueezed.shard_spec == [2, 4, 1, 1]

        # Test inserting dimension with negative index (counting from end)
        unsqueezed = tensor.unsqueeze(-2)
        assert unsqueezed.shape == [10, 20, 1, 30]
        assert unsqueezed.shard_spec == [2, 4, 1, 1]

        # Test out-of-bounds indices
        with pytest.raises(IndexError):
            tensor.unsqueeze(4)
        with pytest.raises(IndexError):
            tensor.unsqueeze(-5)

    def test_squeeze(self):
        """Test squeeze method."""
        # Create tensor with singleton dimensions
        tensor = MetaTensor([1, 10, 1, 20, 1], [1, 2, 1, 4, 1])

        # Test squeezing all singleton dimensions
        squeezed = tensor.squeeze()
        assert squeezed.shape == [10, 20]
        assert squeezed.shard_spec == [2, 4]

        # Test squeezing specific dimension
        squeezed = tensor.squeeze(0)
        assert squeezed.shape == [10, 1, 20, 1]
        assert squeezed.shard_spec == [2, 1, 4, 1]

        # Test squeezing with negative index (counting from end)
        squeezed = tensor.squeeze(-1)
        assert squeezed.shape == [1, 10, 1, 20]
        assert squeezed.shard_spec == [1, 2, 1, 4]

        # Test squeezing non-singleton dimension should raise error
        with pytest.raises(ValueError):
            tensor.squeeze(1)

        # Test out-of-bounds indices
        with pytest.raises(IndexError):
            tensor.squeeze(5)
        with pytest.raises(IndexError):
            tensor.squeeze(-6)

    def test_reshape(self):
        """Test reshape method."""
        tensor = MetaTensor([10, 20, 30], [1, 1, 1])  # Unsharded tensor for simplicity

        # Test basic reshaping
        reshaped = tensor.reshape(20, 300)
        assert reshaped.shape == [20, 300]
        assert reshaped.shard_spec == [1, 1]

        # Test reshaping with inferred dimension
        reshaped = tensor.reshape(50, -1)
        assert reshaped.shape == [50, 120]  # 6000 / 50 = 120
        assert reshaped.shard_spec == [1, 1]

        # Test reshaping that preserves total elements
        reshaped = tensor.reshape(6000)
        assert reshaped.shape == [6000]
        assert reshaped.shard_spec == [1]

        # Test reshape with incompatible dimensions should raise error
        with pytest.raises(ValueError):
            tensor.reshape(100, 100)  # 100*100 != 10*20*30

        # Test reshape with sharded tensor
        sharded_tensor = MetaTensor([10, 20, 30], [2, 1, 1])

        # Test reshaping that preserves the sharded dimension
        # First dimension is 10 with shard 2, so it's divisible (10/2=5 per shard)
        reshaped = sharded_tensor.reshape(10, 600)
        assert reshaped.shape == [10, 600]
        assert reshaped.shard_spec == [2, 1]  # First dimension sharding preserved

        # Test reshaping with dimensions that must be divisible by sharding
        # First dimension split into dimensions that respect sharding
        reshaped = sharded_tensor.reshape(2, 5, 600)  # 2*5*600 = 6000
        assert reshaped.shape == [2, 5, 600]
        # First dimension gets the sharding from the original first dimension
        assert reshaped.shard_spec == [2, 1, 1]  # Only first dimension keeps sharding

        # Test that non-divisible shapes raise errors
        with pytest.raises(ValueError):
            # 5 is not divisible by the shard factor 2
            sharded_tensor.reshape(5, 1200)

        # Test merging multiple sharded dimensions should raise error
        tensor_multi_sharded = MetaTensor([10, 20, 30], [2, 2, 1])
        with pytest.raises(ValueError):
            tensor_multi_sharded.reshape(200, 30)

        # Test reshaping with correctly handling multiple sharded dimensions
        # Both dimensions divisible by their respective shards
        tensor_multi_sharded2 = MetaTensor([4, 6, 30], [2, 2, 1])
        reshaped = tensor_multi_sharded2.reshape(4, 180)  # 4*180 = 4*6*30
        assert reshaped.shape == [4, 180]
        assert reshaped.shard_spec == [2, 2]  # First dimension sharding preserved

    def test_split(self):
        """Test split method."""
        tensor = MetaTensor([10, 20, 30], [1, 2, 1])

        # Test splitting with equal chunks
        splits = tensor.split(5, dim=0)  # Split first dimension into chunks of 5
        assert len(splits) == 2
        assert all(split.shape == [5, 20, 30] for split in splits)
        assert all(split.shard_spec == [1, 2, 1] for split in splits)

        # Test splitting with uneven chunks should raise error
        with pytest.raises(ValueError):
            tensor.split(3, dim=0)  # 10 is not divisible by 3

        # Test splitting with list of sizes
        splits = tensor.split([3, 7], dim=0)
        assert len(splits) == 2
        assert splits[0].shape == [3, 20, 30]
        assert splits[1].shape == [7, 20, 30]
        assert all(split.shard_spec == [1, 2, 1] for split in splits)

        # Test incorrect total size should raise error
        with pytest.raises(ValueError):
            tensor.split([3, 6], dim=0)  # Sum is 9, not 10

    def test_permute(self):
        """Test permute method with various input patterns."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])

        # Test with separate arguments
        permuted = tensor.permute(2, 0, 1)
        assert permuted.shape == [30, 10, 20]
        assert permuted.shard_spec == [1, 2, 4]

        # Test invalid inputs
        with pytest.raises(ValueError):
            tensor.permute(0, 1)  # Too few dimensions

        with pytest.raises(ValueError):
            tensor.permute(0, 0, 1)  # Repeated dimensions

        with pytest.raises(IndexError):
            tensor.permute(0, 1, 3)  # Out of bounds

        # Test negative indices
        permuted = tensor.permute(-1, -3, -2)
        assert permuted.shape == [30, 10, 20]
        assert permuted.shard_spec == [1, 2, 4]

    def test_transpose(self):
        """Test transpose method."""
        tensor = MetaTensor([10, 20, 30, 40], [2, 4, 1, 2])

        # Test transposing arbitrary dimensions
        transposed = tensor.transpose(1, 3)
        assert transposed.shape == [10, 40, 30, 20]
        assert transposed.shard_spec == [2, 2, 1, 4]

        # Test transposing with negative indices
        transposed = tensor.transpose(-4, -2)
        assert transposed.shape == [30, 20, 10, 40]
        assert transposed.shard_spec == [1, 4, 2, 2]

        # Test out-of-bounds indices
        with pytest.raises(IndexError):
            tensor.transpose(0, 4)
        with pytest.raises(IndexError):
            tensor.transpose(-5, 1)

    def test_clone(self):
        """Test clone method."""
        tensor = MetaTensor([10, 20, 30], [2, 4, 1])
        cloned = tensor.clone()

        # Test that clone creates a new object with identical values
        assert cloned is not tensor
        assert cloned.shape == tensor.shape
        assert cloned.shard_spec == tensor.shard_spec

        # Modifying the clone should not affect the original
        cloned._sharded_dims[0] = ShardedDim(15, 3)
        assert tensor.shape[0] == 10
        assert tensor.shard_spec[0] == 2
        assert cloned.shape[0] == 15
        assert cloned.shard_spec[0] == 3

    def test_expand(self):
        """Test expand method."""
        tensor = MetaTensor([1, 20, 1, 30], [1, 4, 1, 1])

        # Test expanding singleton dimensions
        expanded = tensor.expand(10, 20, 15, 30)
        assert expanded.shape == [10, 20, 15, 30]
        assert expanded.shard_spec == [1, 4, 1, 1]  # Preserve sharding where dimensions match

        # Test -1 for unchanged dimensions
        expanded = tensor.expand(-1, -1, 15, -1)
        assert expanded.shape == [1, 20, 15, 30]
        assert expanded.shard_spec == [1, 4, 1, 1]

        # Test trying to expand non-singleton dimension should raise error
        with pytest.raises(ValueError):
            tensor.expand(10, 25, 15, 30)  # Can't expand 20 to 25

        # Test unequal number of dimensions should raise error
        with pytest.raises(ValueError):
            tensor.expand(10, 20, 15)

        # Test with explicit tuple
        expanded = tensor.expand((10, 20, 15, 30))
        assert expanded.shape == [10, 20, 15, 30]
        assert expanded.shard_spec == [1, 4, 1, 1]

    def test_model_id_property(self):
        """Test model_id property getter and setter."""
        # Test default model_id
        tensor = MetaTensor([2, 4, 4])
        assert tensor.model_id == "default"

        # Test custom model_id at initialization
        tensor = MetaTensor([2, 4, 4], model_id="custom_model")
        assert tensor.model_id == "custom_model"

        # Test setter
        tensor.model_id = "new_model"
        assert tensor.model_id == "new_model"

    def test_model_id_preservation_in_operations(self):
        """Test that model_id is preserved during tensor operations."""
        # Create tensor with custom model_id
        tensor = MetaTensor([10, 20, 30], [1, 2, 1], model_id="test_model")

        # Test clone preserves model_id
        cloned = tensor.clone()
        assert cloned.model_id == "test_model"

        # Test reshape preserves model_id
        reshaped = tensor.reshape(10, 600)
        assert reshaped.model_id == "test_model"

        # Test permute preserves model_id
        permuted = tensor.permute(2, 0, 1)
        assert permuted.model_id == "test_model"

        # Test squeeze preserves model_id
        tensor_with_ones = MetaTensor([1, 10, 1], [1, 1, 1], model_id="test_model")
        squeezed = tensor_with_ones.squeeze()
        assert squeezed.model_id == "test_model"

        # Test unsqueeze preserves model_id
        unsqueezed = tensor.unsqueeze(0)
        assert unsqueezed.model_id == "test_model"

    def test_model_id_in_getitem(self):
        """Test that model_id is preserved when using __getitem__."""
        tensor = MetaTensor([10, 20, 30], [1, 2, 1], model_id="test_model")

        # Test slice preserves model_id
        slice_tensor = tensor[1:]
        assert slice_tensor.model_id == "test_model"

        # Verify that individual dimensions don't have model_id
        # This is testing that ShardedDim doesn't try to access model_id
        dim = tensor[0]
        assert isinstance(dim, ShardedDim)

    def test_model_id_in_arithmetic(self):
        """Test that model_id is preserved during arithmetic operations."""

        tensor1 = MetaTensor([10, 20, 30], [1, 2, 1], model_id="test_model")
        tensor2 = MetaTensor([10, 20, 30], [1, 2, 1], model_id="test_model")

        # Test addition preserves model_id
        result = tensor1 + tensor2
        assert result.model_id == "test_model"

        # Test multiplication preserves model_id
        result = tensor1 * tensor2
        assert result.model_id == "test_model"

        # Test scalar operations preserve model_id
        result = tensor1 * 2
        assert result.model_id == "test_model"

    def test_model_id_in_complex_operations(self):
        """Test model_id preservation in more complex operations."""

        tensor = MetaTensor([10, 20, 30], [1, 2, 1], model_id="test_model")

        # Test expand preserves model_id
        expanded = tensor.expand(10, 20, 30)
        assert expanded.model_id == "test_model"

        # Test split preserves model_id
        splits = tensor.split(5, dim=0)
        for split in splits:
            assert split.model_id == "test_model"

        # Test concat preserves model_id
        tensor1 = MetaTensor([10, 20, 30], [1, 2, 1], model_id="test_model")
        tensor2 = MetaTensor([15, 20, 30], [1, 2, 1], model_id="test_model")
        concatenated = tensor1.concat([tensor2], dim=0)
        assert concatenated.model_id == "test_model"

    def teardown_method(self):
        """Clean up after each test."""
        # Reset global state to avoid affecting other tests
        import flagscale.runner.estimator.meta_registry as meta_registry

        meta_registry._model_registries = {}
