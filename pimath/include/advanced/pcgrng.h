// =============================================================================
// PCG random number generator
// @techreport{oneill:pcg2014,
// title = "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation",
// author = "Melissa E. O'Neill",
// institution = "Harvey Mudd College",
// address = "Claremont, CA",
// number = "HMC-CS-2014-0905",
// year = "2014",
// month = Sep,
// xurl = "https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf",
//}
//
// Default values are from pbrt-v3
// https://github.com/mmp/pbrt-v3
//=================================================================================================================
/* pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "../pimath.h"

PIMATH_NAMESPACE_BEGIN

class PCGRNG {
public:
	PCGRNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM){}
    PCGRNG(uint64 sequenceIndex) { setSequence(sequenceIndex); }

    void setSequence(uint64 sequenceIndex) {
        state = 0u;
        inc = (sequenceIndex << 1u) | 1u;
        uniformUInt32();
        state += PCG32_DEFAULT_STATE;
        uniformUInt32();
    }

    PM_INLINE uint32 uniformUInt32() {
        uint64 oldstate = state;
        state = oldstate * PCG32_MULT + inc;
        uint32 xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32 rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    PM_INLINE uint32 uniformUInt32(uint32 b) {
        uint32 threshold = (~b + 1u) % b;
        while (true) {
            uint32 r = uniformUInt32();
            if (r >= threshold) return r % b;
        }
    }

    float32 uniformFloat() {
        return std::min(0.99999994f, float32(uniformUInt32() * 2.3283064365386963e-10f));
    }

    float64 uniformDouble() {
        return std::min(0.99999999999999989, float64(uniformUInt32() * 2.3283064365386963e-10f));
    }

    template<typename T = real, std::enable_if_t<std::is_same<T, float32>::value, int> = 0>
    PM_INLINE real uniformReal() {
        return uniformFloat();
    }

    template<typename T = real, std::enable_if_t<std::is_same<T, float64>::value, int> = 0>
    PM_INLINE real uniformReal() {
        return uniformDouble();
    }
private:
	static constexpr uint64 PCG32_DEFAULT_STATE = 0x853c49e6748fea9bULL;
    static constexpr uint64 PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
    static constexpr uint64 PCG32_MULT = 0x5851f42d4c957f2dULL;

	uint64 state, inc;
};

PIMATH_NAMESPACE_END