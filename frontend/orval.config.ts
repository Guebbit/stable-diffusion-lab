/**
 * Orval config - OpenAPI codegen for the frontend.
 *
 * Generates TypeScript API clients from the backend's openapi.yaml.
 * Run `npm run gen:api` after the backend starts and exports the schema.
 */
import { defineConfig } from 'orval';

export default defineConfig({
  backend: {
    output: {
      mode: 'tags-split',
      target: 'src/api/gen',
      schemas: 'src/api/gen/schemas.ts',
      client: 'react-query',
      baseUrl: 'http://localhost:8000',
      override: {
        query: {
          useQuery: true,
          useInfinite: false,
        },
        mutation: {
          useMutation: true,
        },
        header: () => false,
      },
      urlEncodeParameters: true,
    },
    input: {
      target: '../openapi.yaml',
      override: {
        mutator: {
          path: 'src/api/gen/custom/queryClient.ts',
          name: 'queryClient',
        },
      },
    },
  },
});